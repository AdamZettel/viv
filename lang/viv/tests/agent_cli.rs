// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The command line an agent drives: the
//! `--json` envelope and each command's result schema, the exit codes (one
//! case per code), rule ids and fix edits, `viv help`, `viv map --text`,
//! `viv skill install`, `viv hook post-edit` and the plugin under claude/.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use viv::story::{parse_json, Json};

fn viv_bin() -> &'static str {
    env!("CARGO_BIN_EXE_viv")
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("the repo root")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("viv_agent_cli_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

struct Ran {
    code: i32,
    out: String,
    err: String,
}

/// Run viv in `dir` with stdin closed (no command may wait for input).
fn viv_in(dir: &Path, args: &[&str], env: &[(&str, &str)]) -> Ran {
    let mut c = Command::new(viv_bin());
    c.args(args).current_dir(dir).stdin(Stdio::null()).env_remove("VIV_JSON_STDERR").env_remove("VIV_JSON_RAW");
    for (k, v) in env {
        c.env(k, v);
    }
    let o = c.output().unwrap();
    Ran { code: o.status.code().unwrap_or(-1), out: String::from_utf8_lossy(&o.stdout).into_owned(), err: String::from_utf8_lossy(&o.stderr).into_owned() }
}

/// The envelope a --json command printed: exactly one JSON object on stdout.
fn envelope(r: &Ran) -> Json {
    let j = parse_json(r.out.trim()).unwrap_or_else(|e| panic!("stdout is not one JSON object ({e}):\n{}\n{}", r.out, r.err));
    validate(&j, &schema("envelope"), "envelope");
    assert_eq!(j.get("exit").and_then(Json::num), Some(r.code as f64), "exit in the envelope is the process's: {}", r.out);
    assert_eq!(j.get("ok"), Some(&Json::Bool(r.code == 0)));
    j
}

fn schema(name: &str) -> Json {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("schemas").join(format!("{name}.json"));
    parse_json(&std::fs::read_to_string(&p).unwrap()).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// The part of JSON Schema the schemas use: type, required, properties,
/// items, enum, const, additionalProperties.
fn validate(v: &Json, s: &Json, at: &str) {
    let type_ok = |t: &str| match (t, v) {
        ("object", Json::Obj(_)) | ("array", Json::Arr(_)) | ("string", Json::Str(_)) | ("boolean", Json::Bool(_)) | ("null", Json::Null) => true,
        ("integer", Json::Num(n)) => n.parse::<i64>().is_ok(),
        ("number", Json::Num(_)) => true,
        _ => false,
    };
    match s.get("type") {
        Some(Json::Str(t)) => assert!(type_ok(t), "{at}: expected {t}, found {}", v.text()),
        Some(Json::Arr(ts)) => assert!(ts.iter().filter_map(Json::str).any(type_ok), "{at}: expected one of {}, found {}", Json::Arr(ts.clone()).text(), v.text()),
        _ => {}
    }
    if let Some(Json::Arr(opts)) = s.get("enum") {
        assert!(opts.iter().any(|o| o.text() == v.text()), "{at}: {} is not one of {}", v.text(), Json::Arr(opts.clone()).text());
    }
    if let Some(c) = s.get("const") {
        assert_eq!(c.text(), v.text(), "{at}");
    }
    if let Json::Obj(fields) = v {
        if let Some(Json::Arr(req)) = s.get("required") {
            for r in req.iter().filter_map(Json::str) {
                assert!(fields.iter().any(|(k, _)| k == r), "{at}: missing `{r}` in {}", v.text());
            }
        }
        for (k, x) in fields {
            if let Some(ps) = s.get("properties").and_then(|p| p.get(k)) {
                validate(x, ps, &format!("{at}.{k}"));
            } else if let Some(ap) = s.get("additionalProperties") {
                validate(x, ap, &format!("{at}.{k}"));
            }
        }
    }
    if let (Json::Arr(items), Some(is)) = (v, s.get("items")) {
        for (i, x) in items.iter().enumerate() {
            validate(x, is, &format!("{at}[{i}]"));
        }
    }
}

/// A command's result against its schema (lang/viv/schemas/<cmd>.json).
fn result_ok(env: &Json, cmd: &str) -> Json {
    let r = env.get("result").cloned().unwrap_or(Json::Null);
    validate(&r, &schema(cmd), &format!("{cmd}.result"));
    r
}

fn diags(env: &Json) -> Vec<Json> {
    match env.get("diagnostics") {
        Some(Json::Arr(d)) => d.clone(),
        _ => Vec::new(),
    }
}

fn rules(env: &Json) -> Vec<String> {
    diags(env).iter().filter_map(|d| d.get("rule").and_then(Json::str).map(str::to_string)).collect()
}

const GOOD: &str = "#| one more than x\ndef f(x: Num) -> Num {\n    return x + 1.0\n}\n\nshow f(2.0)\n\ntest \"f adds one\" {\n    assert_near(f(2.0), 3.0, 0.0)\n}\n";

// ------------------------------------------------------------- exit codes

#[test]
fn every_exit_code_has_a_case() {
    let d = scratch("codes");
    std::fs::write(d.join("good.viv"), GOOD).unwrap();
    std::fs::write(d.join("fails.viv"), "test \"two is three\" {\n    assert_near(2.0, 3.0, 0.0)\n}\n").unwrap();
    std::fs::write(d.join("bad.viv"), "def f(x) {\n    return x + y\n}\n").unwrap();
    std::fs::write(d.join("index.viv"), "a = zeros(2)\nshow a[5]\n").unwrap();
    std::fs::write(d.join("loop.viv"), "x = 0.0\nwhile true {\n    x = x + 1.0\n}\n").unwrap();
    let budget = [("VIV_STEP_BUDGET", "100000")];
    let cases: &[(&[&str], &[(&str, &str)], i32, &str)] = &[
        (&["run", "good.viv"], &[], 0, ""),
        (&["run", "fails.viv"], &[], 1, "T-fail"),
        (&["run", "missing.viv"], &[], 2, "U-file"),
        (&["check", "bad.viv"], &[], 3, "E-name-unknown"),
        (&["run", "index.viv"], &[], 4, "R-index"),
        (&["run", "loop.viv"], &budget, 5, "X-limit-steps"),
    ];
    for (args, env, code, rule) in cases {
        // plain mode and --json give the same code
        let plain = viv_in(&d, args, env);
        assert_eq!(plain.code, *code, "viv {args:?}: {}{}", plain.out, plain.err);
        let mut a = args.to_vec();
        a.push("--json");
        let r = viv_in(&d, &a, env);
        assert_eq!(r.code, *code, "viv {a:?}: {}{}", r.out, r.err);
        let e = envelope(&r);
        if !rule.is_empty() {
            assert!(rules(&e).contains(&rule.to_string()), "viv {a:?} should report {rule}: {}", r.out);
        }
    }
    // 6: a policy refusal, here a skill edited since viv wrote it
    let r = viv_in(&d, &["skill", "install", "--json"], &[]);
    assert_eq!(r.code, 0, "{}{}", r.out, r.err);
    std::fs::write(d.join(".claude/skills/vipyr/SKILL.md"), "edited by hand\n").unwrap();
    let r = viv_in(&d, &["skill", "install", "--json"], &[]);
    assert_eq!(r.code, 6, "{}{}", r.out, r.err);
    assert_eq!(rules(&envelope(&r)), vec!["X-policy-overwrite"]);
    // an unknown command is usage
    assert_eq!(viv_in(&d, &["chek", "good.viv"], &[]).code, 2);
    // the table itself
    let r = viv_in(&d, &["help", "exit-codes", "--json"], &[]);
    let e = envelope(&r);
    let codes: Vec<i64> = match e.get("result").and_then(|x| x.get("exit_codes")) {
        Some(Json::Arr(a)) => a.iter().filter_map(|c| c.get("code").and_then(Json::num)).map(|n| n as i64).collect(),
        _ => Vec::new(),
    };
    assert_eq!(codes, vec![0, 1, 2, 3, 4, 5, 6, 70]);
}

// ------------------------------------------------------------- the envelope

#[test]
fn check_and_run_print_the_envelope_on_stdout() {
    let d = scratch("envelope");
    std::fs::write(d.join("good.viv"), GOOD).unwrap();
    let r = viv_in(&d, &["check", "good.viv", "--json"], &[]);
    let e = envelope(&r);
    result_ok(&e, "check");
    assert!(r.err.is_empty(), "nothing on stderr: {}", r.err);
    let r = viv_in(&d, &["run", "good.viv", "--json"], &[]);
    let e = envelope(&r);
    let res = result_ok(&e, "run");
    assert_eq!(res.get("tests").and_then(|t| t.get("pass")).and_then(Json::num), Some(1.0));
    let out = res.get("stdout").and_then(Json::str).unwrap();
    assert!(out.contains("PASS f adds one"), "{out}");
    assert!(!out.contains("__graph__"), "the explorer's lines stay out: {out}");
    // the old form, for one release
    let r = viv_in(&d, &["check", "--json", "missing.viv"], &[("VIV_JSON_STDERR", "1")]);
    assert!(r.out.is_empty());
    assert!(r.err.trim_start().starts_with(r#"{"diagnostics":["#), "{}", r.err);
}

#[test]
fn a_diagnostic_names_the_file_by_its_given_path_with_a_rule() {
    let d = scratch("paths");
    std::fs::write(d.join("bad.viv"), "x = 1.0\nshow x + y\n").unwrap();
    let abs = d.join("bad.viv");
    let r = viv_in(&d, &["check", "--json", abs.to_str().unwrap()], &[]);
    let e = envelope(&r);
    let dg = &diags(&e)[0];
    assert_eq!(dg.get("file").and_then(Json::str), Some(abs.to_str().unwrap()), "{}", r.out);
    assert_eq!(dg.get("rule").and_then(Json::str), Some("E-name-unknown"));
    // every rule a diagnostic carries is in viv help rules
    let r = viv_in(&d, &["help", "rules", "--json"], &[]);
    let listed = r.out.clone();
    for rule in ["E-name-unknown", "H-pow", "H-range", "H-juxtapose", "R-index", "T-fail", "X-limit-steps", "U-file", "C-disagree"] {
        assert!(listed.contains(&format!(r#""id":"{rule}""#)), "{rule} is not in viv help rules");
    }
}

#[test]
fn summary_and_max_lines_bound_a_call() {
    let d = scratch("budget");
    let mut src = String::new();
    for k in 0..30 {
        src.push_str(&format!("show {k}\n"));
    }
    std::fs::write(d.join("many.viv"), &src).unwrap();
    let r = viv_in(&d, &["run", "many.viv", "--json", "--summary"], &[]);
    let e = envelope(&r);
    assert_eq!(e.get("truncated"), Some(&Json::Bool(true)));
    let out = e.get("result").and_then(|x| x.get("stdout")).and_then(Json::str).unwrap().to_string();
    assert_eq!(out.lines().count(), 20, "the last 20 lines: {out}");
    assert!(out.ends_with("29\n"), "{out}");
    let r = viv_in(&d, &["run", "many.viv", "--json", "--max-lines", "0"], &[]);
    assert_eq!(envelope(&r).get("truncated"), Some(&Json::Bool(false)));
    // a list: the features search
    let r = viv_in(&root(), &["features", "--search", "assert", "--json", "--summary"], &[]);
    let e = envelope(&r);
    let res = result_ok(&e, "features");
    match res.get("entries") {
        Some(Json::Arr(a)) => assert!(a.len() <= 5, "{}", r.out),
        _ => panic!("no entries: {}", r.out),
    }
    assert!(res.get("counts").is_some(), "{}", r.out);
}

#[test]
fn the_wrapped_commands_give_their_own_json_as_the_result() {
    let rt = root();
    let r = viv_in(&rt, &["features", "--json", "--max-lines", "0"], &[]);
    let e = envelope(&r);
    assert!(matches!(result_ok(&e, "features").get("entries"), Some(Json::Arr(a)) if a.len() > 200));
    let r = viv_in(&rt, &["explain", "lib/scf/mix.viv", "--json", "--summary"], &[]);
    result_ok(&envelope(&r), "explain");
    let r = viv_in(&rt, &["map", "lib/scf/mix.viv", "--json"], &[]);
    let res = result_ok(&envelope(&r), "map");
    assert!(res.get("modules").map_or(false, |m| !m.text().contains("\"src\"")), "the sources stay out");
    if rt.join("claims").is_dir() {
        let r = viv_in(&rt, &["claims", "--json"], &[]);
        result_ok(&envelope(&r), "claims");
    }
    // VIV_JSON_RAW gives the command's own object
    let r = viv_in(&rt, &["features", "--json"], &[("VIV_JSON_RAW", "1")]);
    assert!(parse_json(r.out.trim()).unwrap().get("areas").is_some());
}

#[test]
fn map_text_puts_each_def_under_its_caller() {
    let r = viv_in(&root(), &["map", "lib/scf/mix.viv", "--text"], &[]);
    assert_eq!(r.code, 0, "{}", r.err);
    let lines: Vec<&str> = r.out.lines().collect();
    assert!(lines[0].starts_with("lib/scf/mix (lib/scf/mix.viv)"), "{}", r.out);
    let at = |name: &str| lines.iter().position(|l| l.trim_start().starts_with(&format!("{name} "))).unwrap_or_else(|| panic!("{name}: {}", r.out));
    let indent = |k: usize| lines[k].len() - lines[k].trim_start().len();
    // pulay is reached from mix_densities, the contract, and sits under it
    let (c, p) = (at("mix_densities"), at("pulay"));
    assert!(lines[c].contains("contract: "), "{}", lines[c]);
    assert!(p > c && indent(p) > indent(c), "{}", r.out);
    let r = viv_in(&root(), &["map", "lib/scf/mix.viv", "nope", "--json"], &[]);
    assert_eq!(r.code, 2, "{}", r.out);
}

// ------------------------------------------------------------- fixes

#[test]
fn hint_fixes_apply_and_the_program_then_checks() {
    let d = scratch("fixes");
    let cases = [
        ("pow.viv", "x = 2.0 ** 3.0\nshow x\n", "H-pow", "x = 2.0 ^ 3.0\n"),
        ("range.viv", "s = 0.0\nfor i in range(4) {\n    s = s + 1.0\n}\nshow s\n", "H-range", "for i to 4 {"),
        ("jux.viv", "x = 3.0\ny = 2x\nshow y\n", "H-juxtapose", "y = 2 * x\n"),
    ];
    for (file, src, rule, want) in cases {
        std::fs::write(d.join(file), src).unwrap();
        let r = viv_in(&d, &["check", file, "--json"], &[]);
        assert_eq!(r.code, 3, "{}", r.out);
        let e = envelope(&r);
        assert_eq!(rules(&e), vec![rule.to_string()], "{}", r.out);
        assert!(diags(&e)[0].get("fix").is_some(), "{rule} carries a fix: {}", r.out);
        std::fs::write(d.join("env.json"), &r.out).unwrap();
        let a = viv_in(&d, &["fix", "--apply-json", "env.json", "--json"], &[]);
        assert_eq!(a.code, 0, "{}{}", a.out, a.err);
        result_ok(&envelope(&a), "fix");
        let now = std::fs::read_to_string(d.join(file)).unwrap();
        assert!(now.contains(want), "{file} after the fix:\n{now}");
        assert_eq!(viv_in(&d, &["check", file], &[]).code, 0, "{now}");
    }
}

// ------------------------------------------------------------- help

#[test]
fn help_works_every_way_an_agent_asks() {
    let d = scratch("help");
    for args in [&["help"][..], &["--help"], &["-h"], &["help", "run"], &["run", "--help"], &["help", "--agent"], &["help", "json"]] {
        let r = viv_in(&d, args, &[]);
        assert_eq!(r.code, 0, "viv {args:?}: {}", r.err);
        assert!(!r.out.is_empty());
    }
    let r = viv_in(&d, &["help", "run", "--json"], &[]);
    let res = result_ok(&envelope(&r), "help");
    assert!(res.get("schema").map_or(false, |s| s.get("$schema").is_some()), "{}", r.out);
    let r = viv_in(&d, &["help", "--agent", "--brief"], &[]);
    assert!(r.out.lines().count() <= 31, "{}", r.out);
    assert!(r.out.contains("vipyr catalog"), "the brief leads with the catalog");
}

/// Every `viv <cmd>` the guide and the skill name is a command `viv help`
/// knows, so neither tells an agent to run something that does not exist.
#[test]
fn the_guide_and_the_skill_name_only_commands_that_exist() {
    let d = scratch("names");
    let r = viv_in(&d, &["help", "--json"], &[]);
    let known: Vec<String> = match envelope(&r).get("result").and_then(|x| x.get("commands")) {
        Some(Json::Arr(a)) => a.iter().filter_map(|c| c.get("name").and_then(Json::str).map(str::to_string)).collect(),
        _ => panic!("{}", r.out),
    };
    let guide = viv_in(&d, &["help", "--agent"], &[]).out;
    let skill = std::fs::read_to_string(root().join("claude/vipyr/skills/vipyr/SKILL.md")).unwrap();
    for (name, text) in [("guide", &guide), ("SKILL.md", &skill)] {
        // a command is `viv x` in backticks, at the start of a line, or in a
        // list after a comma; prose ("every viv command") is not
        for (i, _) in text.match_indices("viv ") {
            let before = &text[..i];
            let line_start = before.rsplit('\n').next().unwrap_or("").trim().is_empty();
            if !(before.ends_with('`') || before.ends_with(", ") || before.ends_with("Bash(") || line_start || before.ends_with("  ")) {
                continue;
            }
            let cmd: String = text[i + 4..].chars().take_while(|c| c.is_ascii_lowercase() || *c == '-').collect();
            if cmd.is_empty() {
                continue;
            }
            assert!(known.contains(&cmd), "the {name} names `viv {cmd}`, which viv help does not list");
        }
    }
}

/// The guide's example program runs and prints what the guide says.
#[test]
fn the_guide_example_runs_as_the_guide_says() {
    let guide = viv_in(&scratch("g0"), &["help", "--agent"], &[]).out;
    let start = guide.find("```viv\n").expect("an example program") + "```viv\n".len();
    let end = start + guide[start..].find("```").unwrap();
    let d = scratch("guide");
    std::fs::write(d.join("sq.viv"), &guide[start..end]).unwrap();
    let r = viv_in(&d, &["run", "sq.viv", "--json"], &[]);
    let e = envelope(&r);
    assert_eq!(r.code, 0, "{}", r.out);
    let out = e.get("result").and_then(|x| x.get("stdout")).and_then(Json::str).unwrap();
    // the lines the guide quotes
    let quoted = guide.split("`viv run sq.viv` prints `").nth(1).and_then(|s| s.split('`').next()).expect("the guide quotes the output");
    assert!(out.contains(quoted), "the guide says `{quoted}`; the run printed:\n{out}");
    let tests = e.get("result").and_then(|x| x.get("tests")).unwrap().text();
    assert!(guide.contains(&tests.replace(",", ", ").replace(":", ": ")), "the guide quotes result.tests as {tests}");
}

// ------------------------------------------------------------- skill, hook, plugin

#[test]
fn skill_install_writes_the_plugin_skill_and_never_settings() {
    let d = scratch("skill");
    let r = viv_in(&d, &["skill", "install", "--json"], &[]);
    let res = result_ok(&envelope(&r), "skill");
    assert!(res.get("allow").is_some());
    let written = std::fs::read_to_string(d.join(".claude/skills/vipyr/SKILL.md")).unwrap();
    assert_eq!(written, std::fs::read_to_string(root().join("claude/vipyr/skills/vipyr/SKILL.md")).unwrap());
    assert!(d.join(".claude/skills/vipyr/reference/house.md").exists());
    assert!(!d.join(".claude/settings.json").exists());
    // again: the same file, no refusal
    assert_eq!(viv_in(&d, &["skill", "install"], &[]).code, 0);
    std::fs::write(d.join(".claude/skills/vipyr/SKILL.md"), "mine\n").unwrap();
    assert_eq!(viv_in(&d, &["skill", "install"], &[]).code, 6);
    assert_eq!(viv_in(&d, &["skill", "install", "--force"], &[]).code, 0);
    let r = viv_in(&d, &["skill", "install", "--plugin"], &[]);
    assert!(r.out.contains("claude plugin marketplace add ./claude"), "{}", r.out);
}

fn hook(dir: &Path, event: &str) -> Ran {
    let mut c = Command::new(viv_bin()).args(["hook", "post-edit"]).current_dir(dir).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    c.stdin.take().unwrap().write_all(event.as_bytes()).unwrap();
    let o = c.wait_with_output().unwrap();
    Ran { code: o.status.code().unwrap_or(-1), out: String::from_utf8_lossy(&o.stdout).into_owned(), err: String::from_utf8_lossy(&o.stderr).into_owned() }
}

#[test]
fn the_post_edit_hook_shows_errors_and_exits_2() {
    let d = scratch("hook");
    std::fs::write(d.join("good.viv"), GOOD).unwrap();
    std::fs::write(d.join("pow.viv"), "x = 2.0 ** 3.0\n").unwrap();
    std::fs::write(d.join("notes.md"), "x ** y\n").unwrap();
    let ev = |f: &str| format!(r#"{{"hook_event_name":"PostToolUse","tool_name":"Edit","cwd":{},"tool_input":{{"file_path":{}}}}}"#, viv::debug_lens::json_str(d.to_str().unwrap()), viv::debug_lens::json_str(&d.join(f).display().to_string()));
    let r = hook(&d, &ev("pow.viv"));
    assert_eq!(r.code, 2, "{}", r.err);
    assert!(r.err.contains("H-pow") && r.err.contains("fix: "), "{}", r.err);
    assert!(r.err.lines().count() <= 16, "{}", r.err);
    assert_eq!(std::fs::read_to_string(d.join("pow.viv")).unwrap(), "x = 2.0 ** 3.0\n", "the hook never rewrites the file");
    assert_eq!(hook(&d, &ev("good.viv")).code, 0);
    assert_eq!(hook(&d, &ev("notes.md")).code, 0);
    assert_eq!(hook(&d, "not json").code, 0);
}

#[test]
fn the_plugin_files_are_well_formed() {
    let c = root().join("claude");
    for f in [".claude-plugin/marketplace.json", "vipyr/.claude-plugin/plugin.json", "vipyr/hooks/hooks.json"] {
        parse_json(&std::fs::read_to_string(c.join(f)).unwrap()).unwrap_or_else(|e| panic!("{f}: {e}"));
    }
    let hooks = std::fs::read_to_string(c.join("vipyr/hooks/hooks.json")).unwrap();
    assert!(hooks.contains("viv hook post-edit"));
    // the skill's frontmatter: description and when_to_use within the
    // 1,536 characters Claude Code reads
    let skill = std::fs::read_to_string(c.join("vipyr/skills/vipyr/SKILL.md")).unwrap();
    let front = skill.split("---").nth(1).unwrap();
    let field = |k: &str| front.lines().find_map(|l| l.strip_prefix(&format!("{k}: "))).unwrap_or("").to_string();
    assert_eq!(field("name"), "vipyr");
    assert!(field("description").len() + field("when_to_use").len() <= 1536);
    assert!(skill.lines().count() < 500);
    // the skill runs nothing without a prompt: every allowed command is read-only
    for t in field("allowed-tools").split(") ").map(|t| t.trim_end_matches(')')) {
        let cmd = t.trim_start_matches("Bash(").split(' ').take(2).collect::<Vec<_>>().join(" ");
        assert!(!["viv run", "viv build", "viv export", "viv fix", "viv skill"].contains(&cmd.as_str()), "{t}");
    }
}
