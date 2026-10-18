// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! `viv atlas <file.viv>`: the atlas of a program, the picture of its defs by
//! module with the calls between them, from the command line.
//!
//! The data is what the site's pages load: the code map of `viv graph --map`
//! (`crate::codemap::map_json`: every def the top level reaches, by module,
//! with its intent, steps, loops and calls), the explorer bundle of `viv
//! graph --bundle` (`crate::graph::bundle_json`), and the call counts of one
//! run of the program as the explorer runs it (the interpreter's `__graph__`
//! line, as www/scripts/explore-build.mjs records them). Three renderings:
//!
//! - [`html`]: one file that opens from file:// with no network. It carries
//!   the map, the counts and the picture `viv graph --svg` draws, beside the
//!   site's own code map component (www/src/components/CodeMap.jsx), bundled
//!   with React by www/scripts/atlas-viewer-build.mjs into `atlas/viewer.js`,
//!   and the walk through the run (the site's components/ZoomWalk.jsx) with
//!   the recording `crate::atlas_walk` makes natively, capped at
//!   WALK_CAP_BYTES.
//! - [`text`]: the modules, each def with its intent line, steps, loops and
//!   call count, and the call tree from the top level.
//! - [`json`]: the bundle with the counts and the map, for agents.

use crate::story::{parse_json, Json};

/// The script and the styles of the page (www/scripts/atlas-viewer-build.mjs).
const VIEWER_JS: &str = include_str!("../atlas/viewer.js");
const VIEWER_CSS: &str = include_str!("../atlas/viewer.css");

/// One run's calls: (caller, callee, how many times), `<main>` for the top level.
pub type Calls = Vec<(String, String, u64)>;

/// The map with each module's path its repository name and `.viv`: the
/// compiler writes the path it was given, which for an absolute one names
/// the machine's directories, and a page that is passed around must not.
pub fn portable(map: &str) -> Result<String, String> {
    let mut j = parse_json(map)?;
    if let Json::Obj(fields) = &mut j {
        for (k, v) in fields.iter_mut() {
            if k != "modules" {
                continue;
            }
            if let Json::Arr(mods) = v {
                for m in mods.iter_mut() {
                    let name = s(m.get("name")).to_string();
                    if let Json::Obj(mf) = m {
                        for (mk, mv) in mf.iter_mut() {
                            if mk == "path" {
                                *mv = Json::Str(format!("{name}.viv"));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(j.text())
}

/// The calls of a run, from its output's `__graph__` line.
pub fn calls_of(output: &str) -> Option<Calls> {
    crate::arch::RunGraph::of_output(output).map(|g| g.edges)
}

/// The calls as JSON: `[[from, to, count], ...]`, the form the site's
/// `<slug>.calls.json` files and CodeMap.jsx read.
pub fn calls_json(calls: &Calls) -> String {
    let rows: Vec<String> = calls.iter().map(|(f, t, c)| format!("[{},{},{c}]", json_str(f), json_str(t))).collect();
    format!("[{}]", rows.join(","))
}

fn json_str(s: &str) -> String {
    crate::debug_lens::json_str(s)
}

/// The bundle, with `"calls"` (null when the program was not run) and
/// `"map"` added: what `viv atlas --json` prints.
pub fn json(bundle: &str, map: &str, calls: Option<&Calls>) -> String {
    let body = bundle.trim_end();
    let body = body.strip_suffix('}').unwrap_or(body);
    let c = calls.map(calls_json).unwrap_or_else(|| "null".into());
    format!("{body},\"calls\":{c},\"map\":{}}}\n", map.trim_end())
}

/// The largest recording a page keeps, in bytes of the story's JSON: past
/// it the page has the atlas and says why it has no walk. (The story itself
/// keeps at most 100,000 events, STORY_CAP in interp.rs.)
pub const WALK_CAP_BYTES: usize = 4_000_000;

/// The walk a page carries: the explorer bundle and the recording of the
/// run (crate::atlas_walk), or why there is none.
pub type WalkData<'a> = Result<(&'a str, &'a crate::atlas_walk::Walk), String>;

/// The walk's part of the page's data: `"walk":{bundle, story, fails, holds,
/// note, seconds}` (what ZoomWalk.jsx's Walk replays) and `"walkNote"`.
pub fn walk_json(walk: &WalkData) -> String {
    match walk {
        Ok((bundle, w)) => format!(
            "\"walk\":{{\"bundle\":{},\"story\":{},\"fails\":{},\"holds\":{},\"note\":{},\"seconds\":{:.2}}},\"walkNote\":null",
            bundle.trim_end(),
            w.rec.story,
            w.rec.fails,
            w.rec.holds,
            w.note.as_deref().map(json_str).unwrap_or_else(|| "null".into()),
            w.seconds
        ),
        Err(why) => format!("\"walk\":null,\"walkNote\":{}", json_str(why)),
    }
}

/// The cap a page keeps: WALK_CAP_BYTES, or VIV_ATLAS_WALK_CAP (bytes).
pub fn walk_cap() -> usize {
    std::env::var("VIV_ATLAS_WALK_CAP").ok().and_then(|v| v.parse().ok()).unwrap_or(WALK_CAP_BYTES)
}

/// Whether a recording fits a page of `cap` bytes: Err says why it does not.
pub fn walk_fits(w: &crate::atlas_walk::Walk, cap: usize) -> Result<(), String> {
    let n = w.rec.story.len();
    if n > cap {
        return Err(format!(
            "No walk: the run recorded {} events ({:.1} MB), more than a page keeps ({:.1} MB). The atlas is the whole program; the explorer on the site walks long runs.",
            w.rec.total,
            n as f64 / 1e6,
            cap as f64 / 1e6
        ));
    }
    Ok(())
}

/// The page: the viewer, and the data in a `#atlas-data` script element.
/// `svg` is the picture, or why there is none.
pub fn html(name: &str, map: &str, calls: Option<&Calls>, svg: Result<&str, &str>, walk: &WalkData) -> String {
    let c = calls.map(calls_json).unwrap_or_else(|| "null".into());
    let (svg_json, err_json) = match svg {
        Ok(s) => (json_str(s), "null".to_string()),
        Err(e) => ("null".to_string(), json_str(e)),
    };
    let data = format!("{{\"name\":{},\"map\":{},\"calls\":{c},\"svg\":{svg_json},\"svgError\":{err_json},{}}}", json_str(name), map.trim_end(), walk_json(walk));
    // `</` would end the script element; `<\/` is the same JSON string
    let data = data.replace("</", "<\\/");
    let title = escape_html(name);
    format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<meta name=\"generator\" content=\"viv atlas\">\n<title>Atlas of {title}</title>\n<style>\n{VIEWER_CSS}\n</style>\n</head>\n<body>\n<div id=\"atlas\"></div>\n<noscript><p class=\"at-none\">The atlas needs JavaScript. `viv atlas {title}.viv --text` prints it.</p></noscript>\n<script type=\"application/json\" id=\"atlas-data\">{data}</script>\n<script>\n{VIEWER_JS}\n</script>\n</body>\n</html>\n"
    )
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// The first sentence of an intent: up to the first ". " (or all of it).
fn first_sentence(s: &str) -> &str {
    let s = s.trim();
    match s.find(". ") {
        Some(k) => &s[..=k],
        None => s,
    }
}

fn arr(j: Option<&Json>) -> &[Json] {
    match j {
        Some(Json::Arr(xs)) => xs,
        _ => &[],
    }
}

fn num(j: Option<&Json>) -> u64 {
    j.and_then(Json::num).map(|x| x as u64).unwrap_or(0)
}

fn s<'a>(j: Option<&'a Json>) -> &'a str {
    j.and_then(Json::str).unwrap_or("")
}

/// The atlas as text: what `viv atlas --text` prints.
pub fn text(map_text: &str, calls: Option<&Calls>) -> Result<String, String> {
    let map = parse_json(map_text)?;
    let modules = arr(map.get("modules"));
    let defs = arr(map.get("defs"));
    let real: Vec<&Json> = defs.iter().filter(|d| !matches!(d.get("main"), Some(Json::Bool(true)))).collect();
    // calls into each def, summed over its callers
    let mut into: std::collections::HashMap<&str, u64> = std::collections::HashMap::new();
    if let Some(cs) = calls {
        for (_, t, c) in cs {
            *into.entry(t.as_str()).or_insert(0) += c;
        }
    }
    let mut out = String::new();
    out.push_str(&format!(
        "atlas of {}: {} defs in {} module{}, reached from the top level ({} defs in the program)\n",
        s(map.get("root")),
        real.len(),
        modules.len(),
        if modules.len() == 1 { "" } else { "s" },
        num(map.get("defsInProgram"))
    ));
    match calls {
        Some(cs) => {
            let total: u64 = cs.iter().map(|c| c.2).sum();
            out.push_str(&format!("run: {total} calls into {} defs (one run, as the explorer runs it)\n", into.len()));
        }
        None => out.push_str("not run: the calls below are the ones the text makes\n"),
    }

    out.push_str("\nmodules\n");
    let width = modules.iter().map(|m| s(m.get("name")).len()).max().unwrap_or(0);
    for (k, m) in modules.iter().enumerate() {
        let n = real.iter().filter(|d| num(d.get("module")) as usize == k).count();
        out.push_str(&format!("  {:<width$}  {:>5} lines  {n} def{}\n", s(m.get("name")), num(m.get("lines")), if n == 1 { "" } else { "s" }));
    }

    out.push_str("\ndefs, by module\n");
    for (k, m) in modules.iter().enumerate() {
        let mut here: Vec<&&Json> = real.iter().filter(|d| num(d.get("module")) as usize == k).collect();
        if here.is_empty() {
            continue;
        }
        here.sort_by_key(|d| num(d.get("line")));
        out.push_str(&format!("  {}\n", s(m.get("name"))));
        for d in here {
            let id = s(d.get("id"));
            let params: Vec<&str> = arr(d.get("params")).iter().filter_map(Json::str).collect();
            let ran = match calls {
                Some(_) => match into.get(id) {
                    Some(c) => format!("  called {c}x"),
                    None => "  not called".into(),
                },
                None => String::new(),
            };
            out.push_str(&format!("    {id}({})  lines {}-{}{ran}\n", params.join(", "), num(d.get("line")), num(d.get("end"))));
            // its `what`, else its doc's first paragraph (as lib/codemap.js's intentOf)
            let doc = s(d.get("doc")).split("\n\n").next().unwrap_or("").split_whitespace().collect::<Vec<_>>().join(" ");
            let what = s(d.get("intent")).trim();
            let intent = first_sentence(if what.is_empty() { &doc } else { what });
            if !intent.is_empty() {
                out.push_str(&format!("        {intent}\n"));
            }
            let steps: Vec<&str> = arr(d.get("steps")).iter().map(|st| s(st.get("id"))).collect();
            if !steps.is_empty() {
                out.push_str(&format!("        steps: {}\n", steps.join(", ")));
            }
            let loops: Vec<String> = arr(d.get("loops")).iter().map(|l| format!("{} (line {})", s(l.get("head")), num(l.get("line")))).collect();
            if !loops.is_empty() {
                out.push_str(&format!("        loops: {}\n", loops.join("; ")));
            }
        }
    }

    out.push_str(&format!("\ncall tree from the top level{}\n", if calls.is_some() { ", with the run's counts" } else { "" }));
    // each def's callees: the run's, else the text's; in the order the text
    // first calls them, then the rest by name
    let static_order = |from: &str| -> Vec<String> {
        let mut v: Vec<(u64, String)> = Vec::new();
        if let Some(d) = defs.iter().find(|d| s(d.get("id")) == from) {
            for c in arr(d.get("calls")) {
                if let Json::Arr(xs) = c {
                    let to = s(xs.first()).to_string();
                    let line = num(xs.get(1));
                    if !to.is_empty() && !v.iter().any(|(_, t)| *t == to) {
                        v.push((line, to));
                    }
                }
            }
        }
        v.sort();
        v.into_iter().map(|(_, t)| t).collect()
    };
    let children = |from: &str| -> Vec<(String, Option<u64>)> {
        let order = static_order(from);
        match calls {
            Some(cs) => {
                let mut v: Vec<(String, Option<u64>)> = cs.iter().filter(|c| c.0 == from && c.1 != from).map(|c| (c.1.clone(), Some(c.2))).collect();
                let rank = |t: &str| order.iter().position(|o| o == t).unwrap_or(usize::MAX);
                v.sort_by(|a, b| rank(&a.0).cmp(&rank(&b.0)).then(a.0.cmp(&b.0)));
                v
            }
            None => order.into_iter().map(|t| (t, None)).collect(),
        }
    };
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    out.push_str("  <main>\n");
    seen.insert("<main>".into());
    let mut stack: Vec<(String, Option<u64>, usize)> = children("<main>").into_iter().rev().map(|(t, c)| (t, c, 1)).collect();
    while let Some((id, count, depth)) = stack.pop() {
        let pad = "  ".repeat(depth + 1);
        let c = count.map(|c| format!("  x{c}")).unwrap_or_default();
        if !seen.insert(id.clone()) {
            out.push_str(&format!("{pad}{id}{c}  (see above)\n"));
            continue;
        }
        let kids = children(&id);
        out.push_str(&format!("{pad}{id}{c}\n"));
        for (t, k) in kids.into_iter().rev() {
            stack.push((t, k, depth + 1));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_adds_the_calls_and_the_map_to_the_bundle() {
        let calls = vec![("<main>".to_string(), "f".to_string(), 3)];
        let j = json("{\"name\":\"p\"}\n", "{\"defs\":[]}", Some(&calls));
        assert_eq!(j, "{\"name\":\"p\",\"calls\":[[\"<main>\",\"f\",3]],\"map\":{\"defs\":[]}}\n");
        assert!(parse_json(&j).is_ok());
        assert!(json("{\"name\":\"p\"}", "{}", None).contains("\"calls\":null"));
    }

    #[test]
    fn the_map_names_modules_by_their_repository_names() {
        let m = portable("{\"modules\":[{\"name\":\"lib/a\",\"path\":\"/home/x/lib/a.viv\",\"lines\":3}]}").unwrap();
        assert_eq!(m, "{\"modules\":[{\"name\":\"lib/a\",\"path\":\"lib/a.viv\",\"lines\":3}]}");
    }

    #[test]
    fn the_page_keeps_its_data_inside_its_script_element() {
        let page = html("p", "{\"defs\":[],\"src\":\"</script>\"}", None, Err("no drawing"), &Err("not run".into()));
        let data = page.split("id=\"atlas-data\">").nth(1).unwrap().split("</script>").next().unwrap();
        assert!(parse_json(data).is_ok(), "{data}");
        assert!(data.contains("<\\/script>"));
    }
}
