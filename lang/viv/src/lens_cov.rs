// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Intent coverage and needed-ness: is
//! every part of a program explained, and is every part needed?
//!
//! - **Coverage** (`viv lens coverage`): per module, how many defs have a
//!   `what`, how many steps have text, how many of the defs' variables have
//!   a glossary line, and how many entries are stale or unstamped. A ratchet
//!   file (`lens_coverage.json` at the repository root) records the numbers,
//!   and `--check` fails when a module's share drops or its count of stale,
//!   dead or broken items grows.
//! - **Needed-ness** (`viv lens needed`): every def must be reachable from
//!   something that runs: the top level or a `test` block of some program or
//!   module in the repository that uses its module, or a def its module's
//!   intent declares an entry point (`entry` in the card, or a `#~ entry`
//!   tag). A function named as a value (a registry member) is reached when
//!   the registry is. Anything else is dead. A def's callers are computed
//!   from the code; what it serves beyond its callers is written, in
//!   `serves [test "..."]` (or `post`, `program`, `hold`, `claim`), and
//!   checked to exist.
//! - **Debt** (`viv debt`): the missing, stale and dead items of each module,
//!   the most central first: defs a story program reaches (the explorer's
//!   programs) before the others, then by how many defs call them.
//!
//! Like lens.rs, nothing here reads the file system: main.rs hands over the
//! texts.

use std::collections::{HashMap, HashSet, VecDeque};

use knotkit::span::SourceId;

use crate::ast::{Item, Stmt};
use crate::lens::{self, Entry, Loaded, OKind};
use crate::parser::MAIN_DEF;

// ---------------------------------------------------------------------------
// Coverage of one module
// ---------------------------------------------------------------------------

/// A share: how many of `total` are done.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Count {
    pub done: usize,
    pub total: usize,
}

impl Count {
    fn add(&mut self, done: bool) {
        self.total += 1;
        if done {
            self.done += 1;
        }
    }
    pub fn plus(self, o: Count) -> Count {
        Count { done: self.done + o.done, total: self.total + o.total }
    }
    /// The share in percent, one decimal; 100 when there is nothing to do.
    pub fn percent(self) -> f64 {
        if self.total == 0 { 100.0 } else { (1000.0 * self.done as f64 / self.total as f64).round() / 10.0 }
    }
    /// Whether this share is below `was`'s (compared exactly, as fractions).
    pub fn below(self, was: Count) -> bool {
        let (a, b) = (self.done as u128 * was.total.max(1) as u128, was.done as u128 * self.total.max(1) as u128);
        if self.total == 0 || was.total == 0 {
            // nothing to do now is never a drop; nothing to do before and
            // something undone now is
            return self.total > 0 && self.done < self.total && was.total == 0;
        }
        a < b
    }
    fn text(self) -> String {
        format!("{}/{}", self.done, self.total)
    }
}

/// One thing a module still owes: a def or step with no intent, a stale or
/// unstamped entry, a dead def, a `serves` target that does not exist.
#[derive(Clone, Debug, PartialEq)]
pub struct Owed {
    /// `what`, `step-text`, `stale`, `unstamped`, `dead`, `serves`
    pub kind: &'static str,
    pub def: String,
    /// `def`, `def/step` or `def:name`
    pub anchor: String,
    pub path: String,
    pub line: usize,
    pub message: String,
}

/// A module's coverage.
#[derive(Clone, Debug, Default)]
pub struct ModCov {
    pub name: String,
    pub path: String,
    /// defs with a `what` (from either home), of all defs but the top level
    pub defs: Count,
    /// steps with text, of all steps the code has (the top level's included)
    pub steps: Count,
    /// parameters and assigned variables with a glossary line, over every def
    pub glossary: Count,
    /// entries whose stamp is not the def's body hash, or that have none
    pub stale: usize,
    /// defs nothing in the repository reaches (filled in from `Needed`)
    pub dead: usize,
    /// `serves` citations whose target does not exist (from `Needed`)
    pub serves: usize,
    /// file entries not signed `by <person>` (unsigned or `by agent`)
    pub unsigned: usize,
    pub items: Vec<Owed>,
}

/// The coverage of the root module of a loaded program.
pub fn coverage(l: &Loaded) -> ModCov {
    let m = &l.root;
    let path = m.text.path.clone();
    let ipath = m.text.intent_path.clone();
    let mut c = ModCov { name: m.text.name.clone(), path: path.clone(), ..Default::default() };
    let (_, entries) = m.entries();
    let entry = |name: &str| -> Option<&Entry> { entries.iter().find(|e| e.name == name) };
    c.unsigned = entries.iter().filter(|e| e.by.as_deref().map_or(true, |b| b.trim() == "agent")).count();

    // the variables of each def need the whole program's facts (which defs
    // write into which arguments); the bodies come from the module's own text
    let whole = crate::parse_keeping_rungs(&l.expanded, SourceId(0)).ok();
    let own = crate::parse_keeping_rungs(&m.text.src, SourceId(0)).ok();
    let cx = whole.as_ref().or(own.as_ref()).map(crate::graph::Cx::of);
    let lines = crate::graph::Lines::new(&m.text.src);

    for d in &m.defs {
        let e = entry(&d.name);
        let mg = lens::merged(d, e);
        let is_main = d.name == MAIN_DEF;
        let at = if d.line > 0 { d.line } else { 1 };
        if !is_main {
            let has_what = mg.field("what").is_some_and(|w| !w.trim().is_empty());
            c.defs.add(has_what);
            if !has_what {
                let (p, line) = match e {
                    Some(e) => (&ipath, e.line),
                    None => (&path, at),
                };
                c.items.push(Owed { kind: "what", def: d.name.clone(), anchor: d.name.clone(), path: p.clone(), line, message: format!("`{}` has no `what`", d.name) });
            }
        }
        let fst = lens::file_step_texts(e);
        for (st, _) in d.steps() {
            let OKind::Step(id) = &st.kind else { continue };
            let has = !st.text.is_empty() || fst.contains_key(id);
            c.steps.add(has);
            if !has {
                c.items.push(Owed { kind: "step-text", def: d.name.clone(), anchor: format!("{}/{id}", d.name), path: path.clone(), line: st.line, message: format!("step `{}/{id}` has no text", d.name) });
            }
        }
        if is_main {
            continue;
        }
        // the glossary: every parameter and every variable the def assigns
        if let (Some(prog), Some(cx)) = (own.as_ref(), cx.as_ref()) {
            if let Some((body, params, _)) = crate::graph::body_of(prog, &d.name) {
                let stmts = crate::graph::doc_statements(&body, cx, &lines);
                let vars = crate::graph::def_variables(&params, &stmts);
                let documented: HashSet<String> = mg.gloss.iter().filter_map(|g| lens::split_doc_key(&g.0).map(|(n, _, _)| n)).collect();
                for v in &vars {
                    c.glossary.add(documented.contains(v));
                }
            }
        }
    }
    // stale and unstamped entries, as `lens check` finds them
    for f in lens::check(l, &mut |_, _| None) {
        if f.kind == "stale" || f.kind == "unstamped" {
            c.stale += 1;
            let def = f.anchor.split(['/', ':']).next().unwrap_or("").to_string();
            c.items.push(Owed { kind: f.kind, def, anchor: f.anchor.clone(), path: f.path.clone(), line: f.line, message: f.message.clone() });
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Needed-ness, over the whole repository
// ---------------------------------------------------------------------------

/// One `.viv` file of the repository, as main.rs reads it.
#[derive(Clone, Debug, Default)]
pub struct RepoFile {
    /// the name `#! use` knows it by: its path under the repository without `.viv`
    pub name: String,
    pub path: String,
    pub src: String,
    pub intent_path: String,
    pub intent: Option<String>,
    /// the modules its `#! use` lines name, as repository names
    pub uses: Vec<String>,
}

/// What a `serves` citation may name, gathered from the repository.
#[derive(Clone, Debug, Default)]
pub struct Known {
    /// every `test "..."` name in every `.viv` file
    pub tests: HashSet<String>,
    /// post slugs: the file names under posts/ without `.viv`
    pub posts: HashSet<String>,
    /// every hold sentence in every `.debug` file
    pub holds: HashSet<String>,
    /// claim sentence ids (posts/claim_verdicts.json, www/src/data/claims.json)
    pub claims: HashSet<String>,
}

/// The kinds a `serves` line may cite.
pub const SERVES_KINDS: [&str; 5] = ["test", "post", "program", "hold", "claim"];

/// The citations of a `serves` text: (kind, target), in order, and whatever
/// else the text holds, which should be nothing.
pub fn serves_targets(text: &str) -> (Vec<(String, String)>, String) {
    let mut out = Vec::new();
    let mut rest = String::new();
    let mut t = text;
    while let Some(k) = t.find('[') {
        rest.push_str(&t[..k]);
        let after = &t[k + 1..];
        let kind: String = after.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
        let after_kind = &after[kind.len()..];
        match after_kind.strip_prefix(" \"").and_then(|a| a.find("\"]").map(|e| (a, e))) {
            Some((a, e)) if SERVES_KINDS.contains(&kind.as_str()) => {
                out.push((kind, a[..e].to_string()));
                t = &a[e + 2..];
            }
            _ => {
                rest.push('[');
                t = after;
            }
        }
    }
    rest.push_str(t);
    (out, rest.trim().to_string())
}

/// A def of the repository, and what is known of its needed-ness.
#[derive(Clone, Debug, Default)]
pub struct DefInfo {
    pub module: String,
    pub name: String,
    pub path: String,
    pub line: usize,
    /// reached from some program's top level or test, or from an entry point
    pub reachable: bool,
    /// declared an entry point
    pub entry: bool,
    /// the defs (`module.def`) that call it or name it, and the files whose
    /// top level or tests do (`module.<main>`)
    pub callers: Vec<String>,
    /// reached from a story program's top level (the explorer's programs)
    pub on_story: bool,
    /// `serves` citations: (kind, target, exists)
    pub serves: Vec<(String, String, bool)>,
}

/// The repository's defs with their needed-ness.
#[derive(Clone, Debug, Default)]
pub struct Needed {
    /// by (module, def)
    pub defs: HashMap<(String, String), DefInfo>,
    /// files that do not parse, with the reason
    pub unparsed: Vec<(String, String)>,
}

impl Needed {
    pub fn get(&self, module: &str, def: &str) -> Option<&DefInfo> {
        self.defs.get(&(module.to_string(), def.to_string()))
    }
    /// The dead defs of a module, in source order.
    pub fn dead_in(&self, module: &str) -> Vec<&DefInfo> {
        let mut v: Vec<&DefInfo> = self.defs.values().filter(|d| d.module == module && !d.reachable).collect();
        v.sort_by_key(|d| d.line);
        v
    }
    /// The broken `serves` citations of a module: (def, kind, target, line).
    pub fn bad_serves_in(&self, module: &str) -> Vec<(&DefInfo, String, String)> {
        let mut v: Vec<(&DefInfo, String, String)> = Vec::new();
        for d in self.defs.values().filter(|d| d.module == module) {
            for (k, t, ok) in &d.serves {
                if !ok {
                    v.push((d, k.clone(), t.clone()));
                }
            }
        }
        v.sort_by_key(|x| x.0.line);
        v
    }
}

/// A parsed file: its defs (name, line, the names each refers to) and the
/// names its top level and tests refer to.
struct Parsed {
    defs: Vec<(String, usize, Vec<String>)>,
    top: Vec<String>,
}

/// The names a statement list refers to that could be defs: every call and
/// every variable read (a def named as a value is a variable read), tests'
/// bodies included.
fn refs_of(body: &[Stmt], names: &HashSet<String>, out: &mut Vec<String>) {
    let cx = crate::graph::Cx { defs: names.clone(), mutates: HashMap::new() };
    fn go(body: &[Stmt], cx: &crate::graph::Cx, out: &mut Vec<String>) {
        for s in body {
            match s {
                Stmt::Test { body, .. } => go(body, cx, out),
                _ => {
                    let mut e = crate::graph::Eff::default();
                    crate::graph::stmt_eff(s, cx, &mut e);
                    for c in e.calls {
                        if !out.contains(&c) {
                            out.push(c);
                        }
                    }
                    // a test inside an `if` or a loop is rare, but its body is code too
                    match s {
                        Stmt::If { then_body, else_body, .. } => {
                            go_tests(then_body, cx, out);
                            go_tests(else_body, cx, out);
                        }
                        Stmt::While { body, .. } | Stmt::For { body, .. } => go_tests(body, cx, out),
                        _ => {}
                    }
                }
            }
        }
    }
    fn go_tests(body: &[Stmt], cx: &crate::graph::Cx, out: &mut Vec<String>) {
        for s in body {
            if let Stmt::Test { body, .. } = s {
                go(body, cx, out);
            }
        }
    }
    go(body, &cx, out);
}

fn parse_file(f: &RepoFile, names: &HashSet<String>) -> Result<Parsed, String> {
    let prog = crate::parse_keeping_rungs(&f.src, SourceId(0)).map_err(|ds| crate::join_errs(&ds))?;
    let lines = crate::graph::Lines::new(&f.src);
    let mut p = Parsed { defs: Vec::new(), top: Vec::new() };
    let mut top: Vec<Stmt> = Vec::new();
    fn items<'a>(its: &'a [Item], out: &mut Vec<&'a Item>) {
        for it in its {
            match it {
                Item::Rung(r) => items(&r.items, out),
                other => out.push(other),
            }
        }
    }
    let mut all = Vec::new();
    items(&prog.items, &mut all);
    for it in all {
        match it {
            Item::Def(d) if crate::update::is_expression_def(&d.name) => {}
            Item::Def(d) => {
                let mut r = Vec::new();
                refs_of(&d.body, names, &mut r);
                                p.defs.push((d.name.clone(), lines.line(d.span.start), r));
            }
            Item::Stmt(s) => top.push(s.clone()),
            _ => {}
        }
    }
    refs_of(&top, names, &mut p.top);
    Ok(p)
}

/// The def names a file's `#~ entry` tags mark, from its source.
fn tagged_entries(src: &str) -> HashSet<String> {
    let sc = crate::scan(src);
    let mut out = HashSet::new();
    let mut tagged_lines: HashSet<usize> = HashSet::new();
    for (l, ts) in &sc.tags_at {
        if ts.iter().any(|t| t == "entry") {
            tagged_lines.insert(*l);
        }
    }
    for sp in &sc.spans {
        if tagged_lines.contains(&sp.start) {
            out.insert(sp.name.clone());
        }
    }
    out
}

/// Needed-ness over every file of the repository. `stories` are the names of
/// the programs the explorer runs.
pub fn needed(files: &[RepoFile], known: &Known, stories: &[String]) -> Needed {
    let mut out = Needed::default();
    // every def name anywhere, so a name read is known to be a def
    let mut names: HashSet<String> = HashSet::new();
    for f in files {
        if let Ok(prog) = crate::parse_keeping_rungs(&f.src, SourceId(0)) {
            fn collect(its: &[Item], names: &mut HashSet<String>) {
                for it in its {
                    match it {
                        Item::Def(d) => {
                            names.insert(d.name.clone());
                        }
                        Item::Rung(r) => collect(&r.items, names),
                        _ => {}
                    }
                }
            }
            collect(&prog.items, &mut names);
        }
    }
    let mut parsed: HashMap<String, Parsed> = HashMap::new();
    for f in files {
        match parse_file(f, &names) {
            Ok(p) => {
                parsed.insert(f.name.clone(), p);
            }
            Err(e) => out.unparsed.push((f.path.clone(), e.lines().next().unwrap_or("").to_string())),
        }
    }
    let by_name: HashMap<&str, &RepoFile> = files.iter().map(|f| (f.name.as_str(), f)).collect();
    // each file's closure: itself, then its uses depth first, each once
    let closure = |root: &str| -> Vec<String> {
        let mut seen: Vec<String> = Vec::new();
        fn visit(n: &str, by: &HashMap<&str, &RepoFile>, seen: &mut Vec<String>) {
            if seen.iter().any(|s| s == n) {
                return;
            }
            seen.push(n.to_string());
            if let Some(f) = by.get(n) {
                for u in &f.uses {
                    visit(u, by, seen);
                }
            }
        }
        visit(root, &by_name, &mut seen);
        seen
    };
    // resolve a name in a closure: the first module of it that defines it
    let resolve = |cl: &[String], name: &str| -> Option<String> {
        cl.iter().find(|m| parsed.get(m.as_str()).is_some_and(|p| p.defs.iter().any(|d| d.0 == name))).cloned()
    };
    let refs_of_def = |m: &str, d: &str| -> Vec<String> {
        parsed.get(m).and_then(|p| p.defs.iter().find(|x| x.0 == d)).map(|x| x.2.clone()).unwrap_or_default()
    };
    for f in files {
        let Some(p) = parsed.get(&f.name) else { continue };
        let entries: HashSet<String> = {
            let mut s = tagged_entries(&f.src);
            if let Some(t) = &f.intent {
                if let Ok(file) = lens::IntentFile::parse(t) {
                    for e in file.model().1 {
                        if e.field("entry").is_some() {
                            s.insert(e.name.clone());
                        }
                    }
                }
            }
            s
        };
        let serves: HashMap<String, Vec<(String, String)>> = match &f.intent {
            Some(t) => match lens::IntentFile::parse(t) {
                Ok(file) => file.model().1.into_iter().filter_map(|e| e.field("serves").map(|s| (e.name.clone(), serves_targets(&s.text).0))).collect(),
                Err(_) => HashMap::new(),
            },
            None => HashMap::new(),
        };
        for (name, line, _) in &p.defs {
            let sv = serves
                .get(name)
                .map(|v| {
                    v.iter()
                        .map(|(k, t)| {
                            let ok = match k.as_str() {
                                "test" => known.tests.contains(t),
                                "post" => known.posts.contains(t.trim_start_matches("posts/").trim_end_matches(".viv")),
                                "program" => by_name.contains_key(t.trim_end_matches(".viv")),
                                "hold" => known.holds.contains(t),
                                "claim" => known.claims.contains(t),
                                _ => false,
                            };
                            (k.clone(), t.clone(), ok)
                        })
                        .collect()
                })
                .unwrap_or_default();
            out.defs.entry((f.name.clone(), name.clone())).or_insert(DefInfo {
                module: f.name.clone(),
                name: name.clone(),
                path: f.path.clone(),
                line: *line,
                entry: entries.contains(name),
                serves: sv,
                ..Default::default()
            });
        }
    }
    // callers: every def's references, resolved in its own module's closure,
    // and every file's top level and tests
    let mut callers: HashMap<(String, String), Vec<String>> = HashMap::new();
    for f in files {
        let Some(p) = parsed.get(&f.name) else { continue };
        let cl = closure(&f.name);
        for (d, _, refs) in &p.defs {
            for r in refs {
                if r == d {
                    continue;
                }
                if let Some(m) = resolve(&cl, r) {
                    let v = callers.entry((m, r.clone())).or_default();
                    let who = format!("{}.{d}", f.name);
                    if !v.contains(&who) {
                        v.push(who);
                    }
                }
            }
        }
        for r in &p.top {
            if let Some(m) = resolve(&cl, r) {
                let v = callers.entry((m, r.clone())).or_default();
                let who = format!("{}.{MAIN_DEF}", f.name);
                if !v.contains(&who) {
                    v.push(who);
                }
            }
        }
    }
    for (k, v) in callers {
        if let Some(d) = out.defs.get_mut(&k) {
            d.callers = v;
        }
    }
    // reachability: from each file's top level and tests in that file's
    // closure, and from each entry point in its module's closure
    let mut reached: HashSet<(String, String)> = HashSet::new();
    let mut story: HashSet<(String, String)> = HashSet::new();
    let walk = |cl: &[String], start: Vec<String>, seen: &mut HashSet<(String, String)>| {
        let mut q: VecDeque<String> = start.into();
        let mut here: HashSet<String> = HashSet::new();
        while let Some(n) = q.pop_front() {
            if !here.insert(n.clone()) {
                continue;
            }
            let Some(m) = resolve(cl, &n) else { continue };
            seen.insert((m.clone(), n.clone()));
            for r in refs_of_def(&m, &n) {
                if !here.contains(&r) {
                    q.push_back(r);
                }
            }
        }
    };
    for f in files {
        let Some(p) = parsed.get(&f.name) else { continue };
        let cl = closure(&f.name);
        walk(&cl, p.top.clone(), &mut reached);
        if stories.iter().any(|s| s == &f.name) {
            walk(&cl, p.top.clone(), &mut story);
        }
    }
    let entry_defs: Vec<(String, String)> = out.defs.values().filter(|d| d.entry).map(|d| (d.module.clone(), d.name.clone())).collect();
    for (m, d) in entry_defs {
        let cl = closure(&m);
        walk(&cl, vec![d], &mut reached);
    }
    for (k, d) in out.defs.iter_mut() {
        d.reachable = reached.contains(k);
        d.on_story = story.contains(k);
    }
    out
}

/// The defs a program's closure defines that its own top level and tests do
/// not reach: informational (another program may need them).
pub fn unreached_by(files: &[RepoFile], program: &str) -> (usize, Vec<(String, String)>) {
    let known = Known::default();
    let only: Vec<RepoFile> = {
        // the program and its closure only
        let by: HashMap<&str, &RepoFile> = files.iter().map(|f| (f.name.as_str(), f)).collect();
        let mut seen: Vec<String> = Vec::new();
        fn visit(n: &str, by: &HashMap<&str, &RepoFile>, seen: &mut Vec<String>) {
            if seen.iter().any(|s| s == n) {
                return;
            }
            seen.push(n.to_string());
            if let Some(f) = by.get(n) {
                for u in &f.uses {
                    visit(u, by, seen);
                }
            }
        }
        visit(program, &by, &mut seen);
        seen.iter().filter_map(|n| by.get(n.as_str()).map(|f| {
            let mut f = (*f).clone();
            // only the program's own top level and tests are roots here
            if f.name != program {
                f.src = strip_top_level(&f.src);
            }
            f.intent = None;
            f
        })).collect()
    };
    let n = needed(&only, &known, &[]);
    let total = n.defs.len();
    let mut v: Vec<(String, String)> = n.defs.values().filter(|d| !d.reachable).map(|d| (d.module.clone(), d.name.clone())).collect();
    v.sort();
    (total, v)
}

/// A module's text with its top-level statements and tests removed, defs
/// and `#!` lines kept, so it defines what it defines and runs nothing.
fn strip_top_level(src: &str) -> String {
    let Ok(prog) = crate::parse_keeping_rungs(src, SourceId(0)) else { return src.to_string() };
    let mut out = String::new();
    for l in src.lines().filter(|l| l.starts_with("#!")) {
        out.push_str(l);
        out.push('\n');
    }
    fn defs(its: &[Item], src: &str, out: &mut String) {
        for it in its {
            match it {
                Item::Def(d) => {
                    out.push_str(&src[d.span.start as usize..d.span.end as usize]);
                    out.push('\n');
                }
                Item::Record(r) => {
                    out.push_str(&src[r.span.start as usize..r.span.end as usize]);
                    out.push('\n');
                }
                Item::Rung(r) => defs(&r.items, src, out),
                _ => {}
            }
        }
    }
    defs(&prog.items, src, &mut out);
    out
}

/// Fill a module's dead and serves counts, and their items, from `Needed`.
pub fn add_needed(c: &mut ModCov, n: &Needed) {
    for d in n.dead_in(&c.name) {
        c.dead += 1;
        c.items.push(Owed {
            kind: "dead",
            def: d.name.clone(),
            anchor: d.name.clone(),
            path: d.path.clone(),
            line: d.line,
            message: format!("`{}` is reached from no program, test or entry point in the repository", d.name),
        });
    }
    for (d, k, t) in n.bad_serves_in(&c.name) {
        c.serves += 1;
        c.items.push(Owed { kind: "serves", def: d.name.clone(), anchor: d.name.clone(), path: d.path.clone(), line: d.line, message: format!("`{}` serves [{k} \"{t}\"], which does not exist", d.name) });
    }
}

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------

fn json_str(s: &str) -> String {
    crate::debug_lens::json_str(s)
}

fn count_json(c: Count) -> String {
    format!("[{},{}]", c.done, c.total)
}

/// One line of the ratchet file, and of `--json`'s modules.
pub fn mod_json(c: &ModCov) -> String {
    format!(
        "{{\"defs\":{},\"steps\":{},\"glossary\":{},\"stale\":{},\"dead\":{},\"serves\":{}}}",
        count_json(c.defs),
        count_json(c.steps),
        count_json(c.glossary),
        c.stale,
        c.dead,
        c.serves
    )
}

pub fn total(cs: &[ModCov]) -> ModCov {
    let mut t = ModCov { name: "total".into(), ..Default::default() };
    for c in cs {
        t.defs = t.defs.plus(c.defs);
        t.steps = t.steps.plus(c.steps);
        t.glossary = t.glossary.plus(c.glossary);
        t.stale += c.stale;
        t.dead += c.dead;
        t.serves += c.serves;
        t.unsigned += c.unsigned;
    }
    t
}

/// The coverage table.
pub fn coverage_text(cs: &[ModCov]) -> String {
    let w = cs.iter().map(|c| c.name.len()).max().unwrap_or(5).max(5);
    let mut s = format!("{:<w$}  {:>13}  {:>13}  {:>15}  {:>5}  {:>4}  {:>6}\n", "module", "defs", "steps", "glossary", "stale", "dead", "serves");
    let row = |c: &ModCov| -> String {
        let cell = |k: Count| format!("{} {:>5}%", k.text(), k.percent());
        format!("{:<w$}  {:>13}  {:>13}  {:>15}  {:>5}  {:>4}  {:>6}\n", c.name, cell(c.defs), cell(c.steps), cell(c.glossary), c.stale, c.dead, c.serves)
    };
    for c in cs {
        s.push_str(&row(c));
    }
    s.push_str(&row(&total(cs)));
    s
}

pub fn coverage_json(cs: &[ModCov]) -> String {
    let mods: Vec<String> = cs
        .iter()
        .map(|c| {
            let items: Vec<String> = c
                .items
                .iter()
                .map(|i| format!("{{\"kind\":\"{}\",\"anchor\":{},\"path\":{},\"line\":{},\"message\":{}}}", i.kind, json_str(&i.anchor), json_str(&i.path), i.line, json_str(&i.message)))
                .collect();
            format!("{{\"module\":{},\"path\":{},\"coverage\":{},\"unsigned\":{},\"items\":[{}]}}", json_str(&c.name), json_str(&c.path), mod_json(c), c.unsigned, items.join(","))
        })
        .collect();
    format!("{{\"modules\":[{}],\"total\":{}}}", mods.join(","), mod_json(&total(cs)))
}

// ---------------------------------------------------------------------------
// The ratchet
// ---------------------------------------------------------------------------

/// A module's recorded numbers.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Recorded {
    pub defs: Count,
    pub steps: Count,
    pub glossary: Count,
    pub stale: usize,
    pub dead: usize,
    pub serves: usize,
}

impl Recorded {
    pub fn of(c: &ModCov) -> Recorded {
        Recorded { defs: c.defs, steps: c.steps, glossary: c.glossary, stale: c.stale, dead: c.dead, serves: c.serves }
    }
}

/// Read the ratchet file: `{"format": 1, "modules": {"lib/hf/rhf": {...}, ...}}`,
/// one module per line as `ratchet_text` writes it.
pub fn parse_ratchet(text: &str) -> Result<Vec<(String, Recorded)>, String> {
    let mut out = Vec::new();
    for (i, l) in text.lines().enumerate() {
        let t = l.trim();
        if !t.starts_with('"') || !t.contains("\"defs\"") {
            continue;
        }
        let name: String = t[1..].chars().take_while(|c| *c != '"').collect();
        let pair = |key: &str| -> Result<Count, String> {
            let at = t.find(&format!("\"{key}\":[")).ok_or_else(|| format!("line {}: `{name}` has no {key}", i + 1))?;
            let rest = &t[at + key.len() + 4..];
            let inner: String = rest.chars().take_while(|c| *c != ']').collect();
            let mut it = inner.split(',').map(|x| x.trim().parse::<usize>());
            match (it.next(), it.next()) {
                (Some(Ok(a)), Some(Ok(b))) => Ok(Count { done: a, total: b }),
                _ => Err(format!("line {}: `{name}` {key} is not [done, total]", i + 1)),
            }
        };
        let num = |key: &str| -> Result<usize, String> {
            let at = t.find(&format!("\"{key}\":")).ok_or_else(|| format!("line {}: `{name}` has no {key}", i + 1))?;
            let rest = &t[at + key.len() + 3..];
            rest.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse::<usize>().map_err(|_| format!("line {}: `{name}` {key} is not a count", i + 1))
        };
        out.push((name.clone(), Recorded { defs: pair("defs")?, steps: pair("steps")?, glossary: pair("glossary")?, stale: num("stale")?, dead: num("dead")?, serves: num("serves")? }));
    }
    Ok(out)
}

/// The ratchet file's text, modules sorted by name.
pub fn ratchet_text(mods: &[(String, Recorded)]) -> String {
    let mut v: Vec<&(String, Recorded)> = mods.iter().collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    let mut s = String::from("{\n\"format\": 1,\n\"about\": \"viv lens coverage --check fails when a module's share of explained defs, steps or glossary terms drops, or its stale, dead or broken-serves count grows. viv lens coverage --update records the current numbers.\",\n\"modules\": {\n");
    for (k, (n, r)) in v.iter().enumerate() {
        s.push_str(&format!(
            "  {}: {{\"defs\":{},\"steps\":{},\"glossary\":{},\"stale\":{},\"dead\":{},\"serves\":{}}}{}\n",
            json_str(n),
            count_json(r.defs),
            count_json(r.steps),
            count_json(r.glossary),
            r.stale,
            r.dead,
            r.serves,
            if k + 1 < v.len() { "," } else { "" }
        ));
    }
    s.push_str("}\n}\n");
    s
}

/// Compare the current numbers with the recorded ones. Returns the drops
/// (each a failure) and the gains (each a hint to `--update`).
pub fn ratchet_compare(name: &str, now: &Recorded, was: &Recorded) -> (Vec<String>, Vec<String>) {
    let mut drops = Vec::new();
    let mut gains = Vec::new();
    for (label, a, b) in [("defs", now.defs, was.defs), ("steps", now.steps, was.steps), ("glossary", now.glossary, was.glossary)] {
        if a.below(b) {
            drops.push(format!("{name}: {label} coverage fell from {} ({}%) to {} ({}%)", b.text(), b.percent(), a.text(), a.percent()));
        } else if b.below(a) {
            gains.push(format!("{name}: {label} coverage rose from {} to {}", b.text(), a.text()));
        }
    }
    for (label, a, b) in [("stale", now.stale, was.stale), ("dead", now.dead, was.dead), ("serves", now.serves, was.serves)] {
        let what = match label {
            "stale" => "stale or unstamped entries",
            "dead" => "dead defs",
            _ => "`serves` citations of nothing",
        };
        if a > b {
            drops.push(format!("{name}: {what} grew from {b} to {a}"));
        } else if a < b {
            gains.push(format!("{name}: {what} fell from {b} to {a}"));
        }
    }
    (drops, gains)
}

// ---------------------------------------------------------------------------
// Debt
// ---------------------------------------------------------------------------

/// The debt of some modules, the most central items first: items of defs a
/// story program reaches, then by callers, then by module and line.
///
/// `run`, when given, is the call count of each def in one recorded run (the
/// interpreter's `__graph__` line, `run_calls`): the defs that run come
/// first, the most called first, before the static order.
pub fn debt_text(cs: &[ModCov], n: &Needed, limit: Option<usize>, run: Option<&HashMap<String, u64>>) -> String {
    let rank = |c: &ModCov, i: &Owed| -> (u64, bool, usize) {
        let calls = run.and_then(|r| r.get(&i.def).copied()).unwrap_or(0);
        match n.get(&c.name, &i.def) {
            Some(d) => (calls, d.on_story, d.callers.len()),
            None => (calls, false, 0),
        }
    };
    let mut s = String::new();
    let t = total(cs);
    s.push_str(&format!(
        "debt: {} defs without a `what`, {} steps without text, {} undocumented variables, {} stale or unstamped entries, {} dead defs, {} `serves` of nothing, {} entries no person has signed\n",
        t.defs.total - t.defs.done,
        t.steps.total - t.steps.done,
        t.glossary.total - t.glossary.done,
        t.stale,
        t.dead,
        t.serves,
        t.unsigned
    ));
    // modules in order of their most central item
    let mut mods: Vec<(&ModCov, (u64, bool, usize))> = cs.iter().filter(|c| !c.items.is_empty()).map(|c| (c, c.items.iter().map(|i| rank(c, i)).max().unwrap_or((0, false, 0)))).collect();
    mods.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.name.cmp(&b.0.name)));
    let mut shown = 0;
    for (c, _) in mods {
        s.push_str(&format!("\n{}  ({} what, {} steps, {} stale, {} dead, {} serves; glossary {})\n", c.name, c.defs.total - c.defs.done, c.steps.total - c.steps.done, c.stale, c.dead, c.serves, c.glossary.text()));
        let mut items: Vec<&Owed> = c.items.iter().collect();
        items.sort_by(|a, b| rank(c, b).cmp(&rank(c, a)).then(a.line.cmp(&b.line)));
        for i in items {
            if limit.is_some_and(|l| shown >= l) {
                s.push_str("  ...\n");
                return s;
            }
            let (calls, story, callers) = rank(c, i);
            let ran = if calls > 0 { format!("ran {calls}x, ") } else { String::new() };
            s.push_str(&format!("  {:<9} {:<40} {ran}{}{} caller{}  {}:{}\n", i.kind, i.anchor, if story { "story, " } else { "" }, callers, if callers == 1 { "" } else { "s" }, i.path, i.line));
            shown += 1;
        }
    }
    s
}

/// The call count of each def in a run, from the interpreter's `__graph__`
/// line (`{"nodes":[...],"edges":[{"from":..,"to":..,"count":..}]}`): the
/// calls into each def, summed over its callers.
pub fn run_calls(output: &str) -> HashMap<String, u64> {
    let mut out = HashMap::new();
    let Some(line) = output.lines().find(|l| l.starts_with("__graph__ ")) else { return out };
    for e in line.split("{\"from\":").skip(1) {
        let to = e.split("\"to\":\"").nth(1).map(|t| t.chars().take_while(|c| *c != '"').collect::<String>());
        let count = e.split("\"count\":").nth(1).map(|t| t.chars().take_while(|c| c.is_ascii_digit()).collect::<String>()).and_then(|c| c.parse::<u64>().ok());
        if let (Some(to), Some(c)) = (to, count) {
            *out.entry(to).or_insert(0) += c;
        }
    }
    out
}
