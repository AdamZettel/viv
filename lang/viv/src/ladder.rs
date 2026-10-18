// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Ladders: a program that builds itself one rung at a time.
//!
//! ```viv
//! rung "Two Gaussians overlap" {
//!     #| The first integral every quantum chemistry code needs.
//!     def overlap(a, b, r2) { ... }
//!     test "S12 for STO-3G hydrogen is 0.6593" { ... }
//! }
//! rung "Kinetic energy, from the overlap" {
//!     def kinetic(a, b, r2) { ... }
//!     test "T11 is 0.7600" { ... }
//! }
//! ```
//!
//! The program at rung k is everything outside any rung (the base) plus rungs
//! 1..k. A `def` in a later rung REPLACES an earlier def of the same name: that
//! is an edit, and the report says so. `viv run` runs the top rung; `viv ladder`
//! runs every rung and holds the ladder to its rules:
//!
//!   1. every test on every rung passes
//!   2. every test of rung k still passes, by name, on rung k + 1
//!   3. every rung after the first adds at least one test
//!   4. a rung that adds nothing -- no def, no statement -- is not a rung
//!   5. a def a rung does not redefine keeps its content address; a def it
//!      does redefine gets a new one (a rung that repeats a body unchanged is
//!      not editing it)
//!
//! Rung texts are cut from the source by span, so a prefix is real source that
//! `viv hash`, `viv doc` and both engines see exactly as a reader does.
use crate::ast::{Item, Program};
use knotkit::span::SourceId;

/// One rung as text.
#[derive(Clone, Debug)]
pub struct RungText {
    pub title: String,
    /// the rung's note: the `#|` lines its body opens with, one per line, a
    /// bare `#|` between paragraphs
    pub note: String,
    /// the body between the braces, dedented one level
    pub text: String,
    /// def names defined in this rung, in order
    pub defs: Vec<String>,
    pub tests: Vec<String>,
    pub line: usize,
}

/// The base (everything before the first rung), the rungs, and the tail
/// (everything after the last rung). Text between two rungs joins the rung
/// before it. The tail runs after the rungs in every prefix, so it is where a
/// ladder puts what its top level does with the finished program.
pub fn split(src: &str, program: &Program) -> (String, Vec<RungText>, String) {
    let mut base = String::new();
    let mut rungs: Vec<RungText> = Vec::new();
    let mut cursor = 0usize;
    for item in &program.items {
        if let Item::Rung(r) = item {
            let between = &src[cursor..r.span.start as usize];
            if let Some(last) = rungs.last_mut() {
                if !between.trim().is_empty() { last.text.push_str(&dedent(between)); }
            } else {
                base.push_str(between);
            }
            let inner = &src[r.body_span.start as usize + 1..r.body_span.end as usize - 1];
            let text = dedent(inner);
            let note = text.lines().take_while(|l| l.trim_start().starts_with("#|"))
                .map(|l| l.trim_start().trim_start_matches('#').trim_start_matches('|').trim()).collect::<Vec<_>>().join("\n");
            let defs = r.items.iter().filter_map(|i| if let Item::Def(d) = i { Some(d.name.clone()) } else { None }).collect();
            let tests = r.items.iter().filter_map(|i| if let Item::Stmt(crate::ast::Stmt::Test { name, .. }) = i { Some(name.clone()) } else { None }).collect();
            rungs.push(RungText { title: r.title.clone(), note, text, defs, tests, line: src[..r.span.start as usize].matches('\n').count() + 1 });
            cursor = r.span.end as usize;
        }
    }
    let tail = src[cursor..].to_string();
    (base, rungs, tail)
}

fn dedent(text: &str) -> String {
    let indent = text.lines().filter(|l| !l.trim().is_empty()).map(|l| l.len() - l.trim_start().len()).min().unwrap_or(0);
    let mut out: String = text.lines().map(|l| if l.len() >= indent { &l[indent..] } else { l.trim_start() }).collect::<Vec<_>>().join("\n");
    out = out.trim_matches('\n').to_string();
    out.push('\n');
    out
}

/// The program at rung k (1-based; 0 is the base alone): base + rungs[..k], with
/// each def replaced by the last rung's version of it.
pub fn prefix(base: &str, rungs: &[RungText], tail: &str, k: usize) -> String {
    let mut text = String::new();
    text.push_str(base.trim_end_matches('\n'));
    text.push('\n');
    for r in &rungs[..k.min(rungs.len())] {
        text.push_str("\n# ---- rung: ");
        text.push_str(&r.title);
        text.push_str(" ----\n");
        // the rung's note is the rung's, not the first def's: keep it as a plain
        // comment so `viv doc` does not hand it to whatever def follows
        let mut in_note = true;
        for line in r.text.lines() {
            if in_note && line.trim_start().starts_with("#|") {
                let note = line.trim_start().trim_start_matches('#').trim_start_matches('|').trim();
                text.push('#');
                if !note.is_empty() { text.push(' '); text.push_str(note); }
            } else {
                in_note = false;
                text.push_str(line);
            }
            text.push('\n');
        }
    }
    if !tail.trim().is_empty() {
        text.push('\n');
        text.push_str(tail.trim_matches('\n'));
        text.push('\n');
    }
    replace_earlier_defs(&text)
}

/// Where a def name appears more than once at the top level, keep the LAST and
/// cut the others out of the text.
pub fn replace_earlier_defs(src: &str) -> String {
    let Ok(program) = crate::parse_keeping_rungs(src, SourceId(0)) else { return src.to_string() };
    let mut last: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let defs: Vec<(&str, usize, usize)> = program.items.iter().filter_map(|i| if let Item::Def(d) = i { Some((d.name.as_str(), d.span.start as usize, d.span.end as usize)) } else { None }).collect();
    for (i, (name, _, _)) in defs.iter().enumerate() {
        last.insert(name, i);
    }
    let mut cuts: Vec<(usize, usize)> = defs.iter().enumerate().filter(|(i, (name, _, _))| last[name] != *i).map(|(_, (_, s, e))| (*s, *e)).collect();
    if cuts.is_empty() {
        return src.to_string();
    }
    cuts.sort();
    let mut out = String::new();
    let mut cursor = 0;
    for (s, e) in cuts {
        // take the def's leading `#|` note lines with it
        let mut start = s;
        while let Some(prev_nl) = src[..start].rfind('\n') {
            let line = &src[prev_nl + 1..start];
            if line.trim_start().starts_with("#|") || line.trim().is_empty() && prev_nl + 1 < start {
                start = prev_nl + 1;
                if line.trim().is_empty() { break; }
            } else { break; }
        }
        out.push_str(&src[cursor..start]);
        cursor = e;
        // and the newline after it
        if src[cursor..].starts_with('\n') { cursor += 1; }
    }
    out.push_str(&src[cursor..]);
    out
}

/// The report `viv ladder` prints, as JSON text: one entry per rung, with the
/// program at that rung, what it added, what it edited, what it carried up with
/// the address it had below, which tests passed and which are new, and every
/// rule it broke. `run` is the host's way of running a program: the output text.
///
/// "Carried" is checked, not assumed: a def the rung did not touch must hash to
/// what it hashed to on the rung below. Content addresses are of the IR, so
/// this holds by construction -- and a compiler bug that made an address depend
/// on where in the file a def sat would show up here, as it once did.
pub fn report(src: &str, run: &mut dyn FnMut(&str) -> Result<String, String>) -> Result<String, String> {
    let program = crate::parse_keeping_rungs(src, SourceId(0)).map_err(|ds| ds.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("\n"))?;
    let (base, rungs, tail) = split(src, &program);
    if rungs.is_empty() {
        return Err("this program has no rungs: a ladder is written `rung \"title\" { ... }`".to_string());
    }
    let mut out = String::from("{\"rungs\":[");
    let mut prev_tests: Vec<String> = Vec::new();
    let mut seen_defs: std::collections::HashMap<String, String> = std::collections::HashMap::new(); // name -> body address
    for (k, r) in rungs.iter().enumerate() {
        let text = prefix(&base, &rungs, &tail, k + 1);
        let output = run(&text)?;
        let passed: Vec<String> = output.lines().filter_map(|l| l.strip_prefix("PASS ")).map(|s| s.to_string()).collect();
        let failed: Vec<String> = output.lines().filter(|l| l.starts_with("FAIL ") || l.starts_with("error")).map(|s| s.to_string()).collect();
        let mut broken: Vec<String> = Vec::new();
        for f in &failed { broken.push(f.clone()); }
        for t in &prev_tests { if !passed.contains(t) { broken.push(format!("the test \"{t}\" from the rung below is gone or renamed")); } }
        if k > 0 && !passed.iter().any(|t| !prev_tests.contains(t)) && failed.is_empty() { broken.push("adds no new test".to_string()); }
        if r.defs.is_empty() && r.tests.is_empty() && r.text.lines().all(|l| l.trim().is_empty() || l.trim_start().starts_with('#')) { broken.push("adds nothing".to_string()); }
        // addresses: which defs of this rung are new, which replace earlier ones,
        // and which earlier ones came up unchanged
        let hashes = crate::hash_report(&text).unwrap_or_default();
        let def_texts: std::collections::HashMap<String, String> = crate::parse_keeping_rungs(&text, SourceId(0)).map(|p| p.items.iter().filter_map(|i| if let Item::Def(d) = i { Some((d.name.clone(), text[d.span.start as usize..d.span.end as usize].to_string())) } else { None }).collect()).unwrap_or_default();
        let addr_of = |name: &str| def_address(&hashes, name, def_texts.get(name).map(|s| s.as_str()));
        let mut added = Vec::new();
        let mut edited = Vec::new();
        let mut carried = Vec::new();
        for (name, old) in seen_defs.iter() {
            if r.defs.contains(name) { continue; }
            let now = addr_of(name);
            if &now != old { broken.push(format!("`{name}` was not redefined on this rung, but its address changed ({old} -> {now})")); }
            carried.push(format!("{{\"name\":{},\"address\":{}}}", json_str(name), json_str(&now)));
        }
        carried.sort();
        for name in &r.defs {
            let addr = addr_of(name);
            match seen_defs.get(name) {
                Some(old) if old == &addr => broken.push(format!("`{name}` is redefined on this rung with the body it already had")),
                Some(old) => edited.push(format!("{{\"name\":{},\"from\":{},\"to\":{}}}", json_str(name), json_str(old), json_str(&addr))),
                None => added.push(format!("{{\"name\":{},\"address\":{}}}", json_str(name), json_str(&addr))),
            }
            seen_defs.insert(name.clone(), addr);
        }
        let new_tests: Vec<&String> = passed.iter().filter(|t| !prev_tests.contains(t)).collect();
        if k > 0 { out.push(','); }
        out.push_str(&format!(
            "{{\"n\":{},\"title\":{},\"note\":{},\"line\":{},\"lines\":{},\"text\":{},\"program\":{},\"added\":[{}],\"edited\":[{}],\"carried\":[{}],\"tests\":[{}],\"newTests\":[{}],\"broken\":[{}],\"output\":{}}}",
            k + 1, json_str(&r.title), json_str(&r.note), r.line, text.lines().count(), json_str(&r.text), json_str(&text),
            added.join(","), edited.join(","), carried.join(","),
            passed.iter().map(|t| json_str(t)).collect::<Vec<_>>().join(","),
            new_tests.iter().map(|t| json_str(t)).collect::<Vec<_>>().join(","),
            broken.iter().map(|t| json_str(t)).collect::<Vec<_>>().join(","),
            json_str(&output.lines().filter(|l| !l.starts_with("__graph__") && !l.starts_with("__trace__") && !l.starts_with("__steps__")).collect::<Vec<_>>().join("\n")),
        ));
        prev_tests = passed;
    }
    out.push_str(&format!("],\"base\":{},\"tail\":{},\"top\":{}}}", json_str(&base), json_str(&tail), json_str(&prefix(&base, &rungs, &tail, rungs.len()))));
    Ok(out)
}

/// A def's body address: from the hash report when the def has a node of its
/// own. A def that takes a function as a parameter has none -- the compiler
/// specializes it per callee (`forward__cube`, `forward__lam0`), and a later
/// rung that calls it with a new function adds a specialization without
/// touching the def. Its address is then a hash of its token stream: still
/// blind to comments and spacing, and the same on every rung that leaves the
/// text alone.
fn def_address(hashes: &str, name: &str, def_text: Option<&str>) -> String {
    if let Some(l) = hashes.lines().find(|l| l.contains(&format!("def {name}("))) {
        return l.split_whitespace().nth(2).unwrap_or("").to_string();
    }
    let Some(text) = def_text else { return String::new() };
    let Ok(toks) = crate::lexer::lex(text, SourceId(0)) else { return String::new() };
    // FNV-1a over the tokens: eight hex digits, like the report's own
    let mut h: u64 = 0xcbf29ce484222325;
    for t in toks {
        for b in format!("{:?} ", t.tok).bytes() { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
    }
    format!("{:08x}", (h as u32) ^ ((h >> 32) as u32))
}

fn json_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
