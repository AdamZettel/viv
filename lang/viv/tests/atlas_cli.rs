// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! `viv atlas` (lang/viv/src/atlas.rs): the text, the JSON and the offline
//! page, on a small program written here and on a story program of six
//! modules (lib/minpack/story_rosenbrock.viv); the page's colored code and
//! the walk it carries (lang/viv/src/atlas_walk.rs), with its cap.

use std::path::{Path, PathBuf};
use std::process::Command;

use viv::story::{parse_json, Json};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("the repo root")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("viv_atlas_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Runs `viv <args>` in `dir`: (status, stdout, stderr).
fn viv(dir: &Path, args: &[&str]) -> (i32, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_viv")).args(args).current_dir(dir).output().expect("viv runs");
    (o.status.code().unwrap_or(-1), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

const SMALL: &str = "# A sum of squares, to map.\n\n# The square of x.\ndef square(x) {\n    return x * x\n}\n\n# The sum of the squares of 0 to n - 1.\ndef total(n) {\n    s = 0.0\n    for k to n {\n        s = s + square(k)\n    }\n    return s\n}\n\nN = 4.0\nprint(total(N))\n";

fn small() -> PathBuf {
    // a directory of its own for each call: the tests run at once
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = scratch(&format!("small{}", N.fetch_add(1, std::sync::atomic::Ordering::SeqCst)));
    std::fs::write(dir.join("squares.viv"), SMALL).unwrap();
    dir
}

/// The calls of a `--json` atlas: (from, to, count).
fn calls(j: &Json) -> Vec<(String, String, u64)> {
    let Some(Json::Arr(rows)) = j.get("calls") else { panic!("no calls") };
    rows.iter()
        .map(|r| {
            let Json::Arr(x) = r else { panic!("a call is not a row") };
            (x[0].str().unwrap().to_string(), x[1].str().unwrap().to_string(), x[2].num().unwrap() as u64)
        })
        .collect()
}

#[test]
fn text_of_a_small_program_counts_the_calls_of_its_run() {
    let dir = small();
    let (code, out, err) = viv(&dir, &["atlas", "squares.viv", "--text"]);
    assert_eq!(code, 0, "{err}");
    assert!(out.starts_with("atlas of squares: 2 defs in 1 module,"), "{out}");
    assert!(out.contains("run: 5 calls into 2 defs"), "{out}");
    assert!(out.contains("    square(x)  lines 4-6  called 4x\n        The square of x.\n"), "{out}");
    assert!(out.contains("        loops: for k to n (line 11)\n"), "{out}");
    assert!(out.contains("  <main>\n    total  x1\n      square  x4\n"), "{out}");
    // with no run, the calls are the text's
    let (code, out, _) = viv(&dir, &["atlas", "squares.viv", "--text", "--no-run"]);
    assert_eq!(code, 0);
    assert!(out.contains("not run: the calls below are the ones the text makes"), "{out}");
    assert!(out.contains("  <main>\n    total\n      square\n"), "{out}");
    assert!(!out.contains("called"), "{out}");
}

#[test]
fn json_of_a_small_program_is_the_bundle_with_the_counts_and_the_map() {
    let dir = small();
    let (code, out, err) = viv(&dir, &["atlas", "squares.viv", "--json"]);
    assert_eq!(code, 0, "{err}");
    let j = parse_json(&out).expect("--json prints JSON");
    for k in ["program", "nodes", "edges", "lines", "modules", "map"] {
        assert!(j.get(k).is_some(), "no {k}");
    }
    let c = calls(&j);
    assert!(c.contains(&("<main>".into(), "total".into(), 1)), "{c:?}");
    assert!(c.contains(&("total".into(), "square".into(), 4)), "{c:?}");
    let (_, out, _) = viv(&dir, &["atlas", "squares.viv", "--json", "--no-run"]);
    assert_eq!(parse_json(&out).unwrap().get("calls"), Some(&Json::Null));
}

#[test]
fn text_and_json_of_a_story_program_of_many_modules() {
    let r = root();
    let (code, out, err) = viv(&r, &["atlas", "lib/minpack/story_rosenbrock.viv", "--text"]);
    assert_eq!(code, 0, "{err}");
    assert!(out.starts_with("atlas of lib/minpack/story_rosenbrock: "), "{out}");
    assert!(out.contains(" in 8 modules"), "{out}");
    for m in ["lib/minpack/lmder", "lib/minpack/lmpar", "lib/minpack/qrfac", "lib/minpack/qrsolv", "lib/minpack/enorm", "lib/minpack/mgh"] {
        assert!(out.contains(&format!("\n  {m} ")), "no module {m}: {out}");
    }
    // the tree goes from the top level through lmder1 to lmder and its solvers
    let tree = out.split("call tree from the top level").nth(1).expect("a call tree");
    for d in ["    lmder1  x1\n", "      lmder  x1\n", "        qrfac  x", "        lmpar  x", "          qrsolv  x"] {
        assert!(tree.contains(d), "no `{}` in the tree: {tree}", d.trim());
    }
    assert!(out.contains("steps: setup, check_input, start, jacobian, qr"), "{out}");

    let (code, out, err) = viv(&r, &["atlas", "lib/minpack/story_rosenbrock.viv", "--json"]);
    assert_eq!(code, 0, "{err}");
    let j = parse_json(&out).unwrap();
    let c = calls(&j);
    assert!(c.iter().any(|(f, t, n)| f == "lmder" && t == "lmpar" && *n > 0), "{c:?}");
    let Some(Json::Arr(mods)) = j.get("map").and_then(|m| m.get("modules")) else { panic!("no map modules") };
    assert_eq!(mods.len(), 8);
}

#[test]
fn the_page_carries_its_data_and_viewer_and_loads_nothing() {
    let r = root();
    let dir = scratch("page");
    let page = dir.join("rosen.atlas.html");
    // an absolute path: the page must still name modules by repository names
    let file = r.join("lib/minpack/story_rosenbrock.viv");
    let (code, out, err) = viv(&dir, &["atlas", file.to_str().unwrap(), "-o", page.to_str().unwrap()]);
    assert_eq!(code, 0, "{err}");
    assert!(out.starts_with("wrote "), "{out}");
    let html = std::fs::read_to_string(&page).unwrap();
    assert!(html.starts_with("<!doctype html>"));
    assert!(html.contains("<title>Atlas of lib/minpack/story_rosenbrock</title>"));
    // the data: the map, the run's counts and the picture
    let data = html.split("<script type=\"application/json\" id=\"atlas-data\">").nth(1).unwrap().split("</script>").next().unwrap();
    let j = parse_json(data).expect("the page's data is JSON");
    assert!(matches!(j.get("calls"), Some(Json::Arr(c)) if !c.is_empty()));
    assert!(j.get("svg").and_then(Json::str).is_some_and(|s| s.contains("<svg")), "no picture");
    assert!(j.get("map").and_then(|m| m.get("defs")).is_some());
    // the viewer is inline: no script, stylesheet or font from anywhere
    assert!(html.contains("createRoot") || html.contains("atlas-data"), "no viewer");
    for bad in ["<script src", "<link", "@import", "new Worker"] {
        assert!(!html.contains(bad), "the page contains {bad}");
    }
    // a `url(` names a marker of the page's own SVG (`url(#cm-arrow-...)`)
    // or holds a KaTeX font written into the page
    for (k, _) in html.match_indices("url(") {
        let rest = &html[k + "url(".len()..];
        assert!(rest.starts_with('#') || rest.starts_with("data:font/woff2;base64,"), "a url( in the page: {}", &html[k..(k + 60).min(html.len())]);
    }
    // every URL in it is an XML namespace, which names and loads nothing
    for (k, _) in html.match_indices("http") {
        let rest = &html[k..];
        if rest.starts_with("http://") || rest.starts_with("https://") {
            assert!(rest.starts_with("http://www.w3.org/"), "a URL in the page: {}", &rest[..rest.len().min(80)]);
        }
    }
    // and no directory of this machine
    let here = r.to_string_lossy().to_string();
    assert!(!html.contains(&here), "the page names {here}");
}

#[test]
fn atlas_without_a_file_prints_its_usage() {
    let (code, _, err) = viv(&root(), &["atlas"]);
    assert_ne!(code, 0);
    assert!(err.contains("usage: viv atlas <file.viv>"), "{err}");
    let (code, _, err) = viv(&root(), &["atlas", "lib/minpack/story_rosenbrock.viv", "--nope"]);
    assert_ne!(code, 0);
    assert!(err.contains("no option --nope"), "{err}");
}

const FOREVER: &str = "def step1(x) {\n    return x + 1.0\n}\n\nx = 0.0\nwhile true {\n    x = step1(x)\n}\n";

#[test]
fn a_run_that_never_ends_stops_at_the_step_budget_and_the_atlas_is_the_texts() {
    let dir = scratch("forever");
    std::fs::write(dir.join("forever.viv"), FOREVER).unwrap();
    // a small budget keeps the test quick; the default is main.rs ATLAS_STEP_BUDGET
    let budget = "100000";
    let o = Command::new(env!("CARGO_BIN_EXE_viv"))
        .args(["atlas", "forever.viv", "--text"])
        .env("VIV_STEP_BUDGET", budget)
        .current_dir(&dir)
        .output()
        .expect("viv runs");
    let (out, err) = (String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
    assert_eq!(o.status.code(), Some(0), "{err}");
    assert!(err.starts_with("viv atlas: running forever.viv once for its call counts (--no-run skips the run)\n"), "{err}");
    assert!(err.contains("step budget exceeded"), "{err}");
    assert!(out.starts_with("atlas of forever: 1 defs in 1 module,"), "{out}");
    assert!(out.contains("  <main>\n    step1\n"), "{out}");
}

/// The data of a page `viv atlas` wrote.
fn page_data(html: &str) -> Json {
    let data = html.split("<script type=\"application/json\" id=\"atlas-data\">").nth(1).unwrap().split("</script>").next().unwrap();
    parse_json(&data.replace("<\\/", "</")).expect("the page's data is JSON")
}

/// `viv atlas <file> -o <dir>/<tag>.atlas.html` with `env`: (page, stderr).
fn page_of(dir: &Path, file: &str, tag: &str, extra: &[&str], env: &[(&str, &str)]) -> (String, String) {
    let page = scratch(tag).join(format!("{tag}.atlas.html"));
    let mut args = vec!["atlas", file, "-o", page.to_str().unwrap()];
    args.extend_from_slice(extra);
    let o = Command::new(env!("CARGO_BIN_EXE_viv")).args(&args).envs(env.iter().copied()).current_dir(dir).output().expect("viv runs");
    let err = String::from_utf8_lossy(&o.stderr).into_owned();
    assert_eq!(o.status.code(), Some(0), "{err}");
    (std::fs::read_to_string(&page).unwrap(), err)
}

#[test]
fn the_page_colors_code_with_the_sites_viv_tokens_in_light_and_dark() {
    let (html, _) = page_of(&small(), "squares.viv", "colors", &["--no-run"], &[]);
    let style = html.split("<style>").nth(1).unwrap().split("</style>").next().unwrap();
    // the rules for the classes lib/highlight.js emits, from styles/viv.css
    for (cls, var) in [("kw", "kw"), ("com", "com"), ("str", "str"), ("num", "num"), ("fn", "fn"), ("call", "call"), ("type", "type"), ("op", "op")] {
        assert!(style.contains(&format!(".tk-{cls}")) && style.contains(&format!("color: var(--tk-{var})")), "no rule for .tk-{cls}");
    }
    // each token's color is set for light, and again for a dark system
    // (in src/atlas/atlas.css, which follows the data palette's own two schemes)
    let atlas_css = &style[style.find("--radius: 10px;").expect("atlas.css in the page")..];
    let (light, dark) = atlas_css.split_once("@media (prefers-color-scheme: dark)").expect("a dark scheme");
    for v in ["--tk-kw:", "--tk-com:", "--tk-str:", "--tk-num:", "--tk-fn:", "--tk-type:"] {
        assert!(light.contains(v) && dark.contains(v), "{v} is not set for both schemes");
    }
    // the viewer emits the classes, and knows Viv's keywords and builtins
    let script = html.split("</script>\n<script>\n").nth(1).expect("the viewer's script");
    assert!(script.contains("class=\"tk-"), "the viewer emits no token classes");
    for w in ["\"while\"", "\"return\"", "\"assert_near\""] {
        assert!(script.contains(w), "the highlighter does not know {w}");
    }
}

#[test]
fn the_page_typesets_math_with_katex_and_its_fonts_inline() {
    let (html, _) = page_of(&small(), "squares.viv", "katex", &["--no-run"], &[]);
    let style = html.split("<style>").nth(1).unwrap().split("</style>").next().unwrap();
    let script = html.split("</script>\n<script>\n").nth(1).expect("the viewer's script");
    // KaTeX is in the bundle (its renderer and its parser's errors), not left out
    assert!(script.contains("renderToString") && script.contains("KaTeX parse error"), "no KaTeX in the viewer");
    // its stylesheet is in the page, and every face it names is a woff2 in it
    assert!(style.contains(".katex{") || style.contains(".katex {"), "no KaTeX stylesheet");
    let faces: Vec<&str> = style.split("@font-face{").skip(1).map(|f| f.split('}').next().unwrap()).filter(|f| f.contains("KaTeX_")).collect();
    assert!(faces.len() >= 10, "{} KaTeX fonts in the page", faces.len());
    for f in &faces {
        assert!(f.contains("src:url(data:font/woff2;base64,"), "a KaTeX font not inline: {}", &f[..f.len().min(120)]);
        assert!(!f.contains(".woff)") && !f.contains(".ttf)") && !f.contains("fonts/"), "a KaTeX font loads a file: {}", &f[..f.len().min(120)]);
    }
    // no font, stylesheet or script from anywhere
    for bad in ["fonts/KaTeX", "cdn.jsdelivr", "katex.org", ".woff2)", "@import"] {
        assert!(!html.contains(bad), "the page contains {bad}");
    }
}

#[test]
fn the_page_carries_the_walk_recorded_by_the_same_run_natively() {
    let r = root();
    let (html, err) = page_of(&r, "lib/minpack/story_rosenbrock.viv", "walk", &[], &[]);
    assert!(err.contains("viv atlas: recording the walk"), "{err}");
    let j = page_data(&html);
    assert_eq!(j.get("walkNote"), Some(&Json::Null));
    let w = j.get("walk").expect("a walk");
    // the explorer bundle the walk lays out, and the run's story
    assert!(w.get("bundle").and_then(|b| b.get("program")).and_then(Json::str).is_some_and(|p| p.contains("def lmder(")));
    // with the run's calls, as explore-build.mjs's bundle has them (the
    // walk's parallelReached follows them)
    assert!(
        matches!(w.get("bundle").and_then(|b| b.get("calls")), Some(Json::Arr(rows)) if rows.iter().any(|r| matches!(r, Json::Arr(x) if x.first().and_then(Json::str) == Some("lmder1") && x.get(1).and_then(Json::str) == Some("lmder")))),
        "the walk's bundle has no calls"
    );
    let Some(Json::Arr(events)) = w.get("story").and_then(|s| s.get("events")) else { panic!("no events") };
    let total = w.get("story").and_then(|s| s.get("total")).and_then(Json::num).unwrap() as usize;
    assert_eq!(total, events.len());
    let kinds = |k: &str, d: &str| events.iter().filter(|e| matches!(e, Json::Arr(x) if x[0].str() == Some(k) && x[1].str() == Some(d))).count();
    // every def traced, so a def without steps (enorm) has its calls too
    assert!(kinds("call", "lmder") == 1 && kinds("call", "enorm") > 1 && kinds("call", "qrsolv") > 0, "not every def traced");
    // and lines, of every pass (a short run is recorded whole)
    let lines = events.iter().filter(|e| matches!(e, Json::Arr(x) if x[0].str() == Some("watch") && x[2].str().is_some_and(|a| a.starts_with("line:")))).count();
    assert!(lines > 1000, "{lines} lines recorded");
    assert_eq!(w.get("note"), Some(&Json::Null));
    assert!(html.len() < 3_000_000, "the page is {} bytes", html.len());
    assert!(!html.contains(&r.to_string_lossy().to_string()), "the page names this machine's directories");
}

#[test]
fn a_recording_past_the_cap_leaves_the_atlas_without_a_walk() {
    let dir = small();
    // a cap no recording fits
    let (html, err) = page_of(&dir, "squares.viv", "cap", &[], &[("VIV_ATLAS_WALK_CAP", "50")]);
    let j = page_data(&html);
    assert_eq!(j.get("walk"), Some(&Json::Null));
    let note = j.get("walkNote").and_then(Json::str).expect("a note");
    assert!(note.starts_with("No walk: the run recorded ") && note.contains("more than a page keeps"), "{note}");
    assert!(err.contains(note), "{err}");
    // the atlas is all there
    assert!(matches!(j.get("calls"), Some(Json::Arr(c)) if !c.is_empty()));
    // with the default cap the walk is there
    let (html, _) = page_of(&dir, "squares.viv", "nocap", &[], &[]);
    assert!(matches!(page_data(&html).get("walk"), Some(Json::Obj(_))));
    // asked for none, or with no run, there is none, and the page says why
    for (flag, why) in [("--no-walk", "(--no-walk)"), ("--no-run", "(--no-run)")] {
        let (html, _) = page_of(&dir, "squares.viv", flag.trim_start_matches('-'), &[flag], &[]);
        let j = page_data(&html);
        assert_eq!(j.get("walk"), Some(&Json::Null));
        assert!(j.get("walkNote").and_then(Json::str).is_some_and(|n| n.contains(why)), "{flag}");
    }
}

#[test]
fn a_second_recording_past_the_cap_leaves_the_first() {
    // the Skyrme run's every pass line by line is about 26 MB; the page keeps
    // the first recording, each loop's first pass line by line, and says so
    let (html, _) = page_of(&root(), "lib/skyrme/story_o16.viv", "o16", &[], &[]);
    let j = page_data(&html);
    let note = j.get("walk").and_then(|w| w.get("note")).and_then(Json::str).expect("a note on the walk");
    assert!(note.starts_with("The passes after the first are not recorded line by line"), "{note}");
    assert!(html.len() < 3_000_000, "the page is {} bytes", html.len());
}
