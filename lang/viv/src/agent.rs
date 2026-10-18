// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The parts of the command line an agent relies on: `viv help`, the JSON envelope, rule ids and fix edits on
//! diagnostics, the exit codes, `viv map --text`, `viv skill install` and
//! `viv hook post-edit`.
//!
//! With `--json`, a command prints one object on stdout:
//!
//!     {"viv":1,"cmd":"check","ok":false,"exit":3,"diagnostics":[...],
//!      "result":{...},"truncated":false,"more":null}
//!
//! `check` and `run` build it themselves (main.rs). `features`, `explain`,
//! `claims` and `map` keep their own JSON and are wrapped here: viv runs
//! itself again with VIV_JSON_RAW=1, which gives the command's own object,
//! and that object becomes `result`. VIV_JSON_STDERR=1 keeps the shape and
//! stream `check --json` and `run --json` had before (one release).

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

use viv::debug_lens::json_str;
use viv::errkind::ErrKind;
use viv::story::{parse_json, Json};

/// The envelope's schema version: a field added keeps it, a field removed or
/// renamed bumps it.
pub const SCHEMA: u32 = 1;

pub const OK: u8 = 0;
pub const TEST_FAILED: u8 = 1;
pub const USAGE: u8 = 2;
pub const COMPILE: u8 = 3;
pub const RUNTIME: u8 = 4;
pub const LIMIT: u8 = 5;
pub const POLICY: u8 = 6;
pub const INTERNAL: u8 = 70;

/// Every exit code, with what it means (`viv help exit-codes`).
pub const EXIT_CODES: &[(u8, &str, &str)] = &[
    (OK, "ok", "done; every test passed"),
    (TEST_FAILED, "test-failed", "the program ran and a test, hold or contract agreement failed (T-fail, T-hold, C-disagree)"),
    (USAGE, "usage", "unknown command or flag, missing file, bad --set"),
    (COMPILE, "compile", "the program does not compile (E- and H- diagnostics)"),
    (RUNTIME, "runtime", "a run-time error (R- diagnostics; the kind is in the rule)"),
    (LIMIT, "limit", "a limit stopped it (X-limit-steps)"),
    (POLICY, "policy", "refused: viv will not do it without a flag (X-policy-*), such as overwriting an edited skill"),
    (INTERNAL, "internal", "an internal error in viv: a bug, please report it"),
];

/// Every rule id a diagnostic can carry: (id, what it means, an example).
pub const RULES: &[(&str, &str, &str)] = &[
    ("E-lex", "a character, number, string or escape the lexer cannot read", "s = \"open"),
    ("E-syntax", "the parser expected something else here", "x = (1 +"),
    ("E-name-unknown", "a variable that is not a parameter, local or global in scope", "return x + y   (no y)"),
    ("E-call-unknown", "a call to a function no def, import or builtin provides", "y = frobnicate(2)"),
    ("E-type-unknown", "a type or record name that is not defined", "def f(p: Vec3) { ... }"),
    ("E-field", "a record has no such field, or an enum no such variant", "p.z   (record P has x, y)"),
    ("E-arity", "a call with the wrong number of arguments", "sqrt(1, 2)"),
    ("E-keyword", "an unknown keyword argument, or one given twice", "sum(X, axs=0)"),
    ("E-redefined", "a function or record defined twice", "def f() { ... } def f() { ... }"),
    ("E-match", "a match that misses a case, or an arm that can never run", "match c { Red => 1 }"),
    ("E-check", "any other front-end check (types, shapes, scopes)", "x = [1, 2] + \"a\""),
    ("H-pow", "`**` from Python: Viv writes x ^ y", "x = 2 ** 3"),
    ("H-range", "`range` from Python: Viv loops with for i to n", "for i in range(4) { ... }"),
    ("H-array-cond", "an array comparison where a condition goes: use all() or any()", "if a == b { ... }   (arrays)"),
    ("H-juxtapose", "a number written against a name: Viv has no implicit multiplication", "y = 2x"),
    ("R-error", "a run-time error with no more specific kind", "stop(\"no root in [a, b]\")"),
    ("R-index", "an index outside an array", "y[5] on an array of 2"),
    ("R-value", "a value outside what the operation takes", "sqrt(-1.0) on a real"),
    ("R-overflow", "an Int result that does not fit", "Int max + 1"),
    ("R-zerodiv", "an Int division by zero", "7 // 0"),
    ("R-recursion", "recursion deeper than the limit every engine shares", "def f(n) { return f(n + 1) }"),
    ("R-memory", "an allocation that cannot be made", "zeros(1e15)"),
    ("R-interrupt", "the run was interrupted", "Ctrl-C"),
    ("T-fail", "a test assertion failed", "assert_near(f(2), 4.0, 1e-12) when f(2) is 3"),
    ("T-hold", "a hold in a debug lens failed", "hold residual < 1e-10"),
    ("C-disagree", "a contract's methods disagree by more than its agree tolerance", "using every method"),
    ("X-limit-steps", "the interpreter's step budget ran out (VIV_STEP_BUDGET raises it)", "while true { }"),
    ("X-policy-overwrite", "viv skill install will not replace a SKILL.md edited since viv wrote it (--force does)", "viv skill install"),
    ("S11-options", "options passed by position that can be named (viv fix --keywords; a note, not a failure)", "sum(X, -1)"),
    ("U-file", "an input file that cannot be read", "viv check missing.viv"),
    ("U-set", "a --set or --using that does not apply", "viv run --set NOPE=1 f.viv"),
    ("U-usage", "an unknown command or flag", "viv chek f.viv"),
    ("I-internal", "viv stopped unexpectedly; a bug", ""),
];

/// The exit code for a rule.
pub fn exit_for_rule(rule: &str) -> u8 {
    match rule {
        "T-fail" | "T-hold" | "C-disagree" => TEST_FAILED,
        r if r.starts_with("U-") => USAGE,
        r if r.starts_with("E-") || r.starts_with("H-") => COMPILE,
        r if r.starts_with("R-") => RUNTIME,
        r if r.starts_with("X-limit") => LIMIT,
        r if r.starts_with("X-policy") => POLICY,
        r if r.starts_with('S') || r.starts_with("L-") => OK,
        _ => INTERNAL,
    }
}

/// The worst exit code of a set of rules: the order of the exit codes puts
/// the more basic failure first (usage before
/// compile before run time before a failed test).
pub fn exit_for_rules<'a>(rules: impl Iterator<Item = &'a str>) -> u8 {
    let rank = |c: u8| match c {
        USAGE => 0,
        COMPILE => 1,
        INTERNAL => 2,
        LIMIT => 3,
        POLICY => 4,
        RUNTIME => 5,
        TEST_FAILED => 6,
        _ => 7,
    };
    rules.map(exit_for_rule).min_by_key(|c| rank(*c)).unwrap_or(OK)
}

/// The rule of a diagnostic, from its message, its notes and, for a run-time
/// error, the kind errkind.rs recorded.
pub fn rule_of(message: &str, notes: &[String], kind: Option<ErrKind>) -> &'static str {
    let m = message;
    if m.contains("step budget exceeded") {
        return "X-limit-steps";
    }
    if m.contains("contract ") && m.contains(" disagrees with ") {
        return "C-disagree";
    }
    if m.ends_with("FAILED") && m.contains("assertion") {
        return "T-fail";
    }
    if m.ends_with("FAILED") && m.contains("hold") {
        return "T-hold";
    }
    if let Some(k) = kind {
        return match k {
            ErrKind::Index => "R-index",
            ErrKind::Value => "R-value",
            ErrKind::Overflow => "R-overflow",
            ErrKind::ZeroDiv => "R-zerodiv",
            ErrKind::Recursion => "R-recursion",
            ErrKind::Memory => "R-memory",
            ErrKind::Interrupt => "R-interrupt",
            ErrKind::Error | ErrKind::Stop => "R-error",
        };
    }
    if m.starts_with("cannot read") || m.contains(": cannot read ") {
        return "U-file";
    }
    for n in notes {
        if n == viv::hints::POW {
            return "H-pow";
        }
        if n == viv::hints::FOR_IN || n == viv::hints::RANGE {
            return "H-range";
        }
        if n == viv::hints::ARRAY_CONDITION {
            return "H-array-cond";
        }
        if n.starts_with("Viv has no implicit multiplication") {
            return "H-juxtapose";
        }
    }
    let has = |w: &str| m.contains(w);
    if has("unterminated") || has("invalid number") || has("unknown escape") || has("unexpected character") || has("NUL character") {
        "E-lex"
    } else if m.starts_with("expected") || m.starts_with("unexpected") {
        "E-syntax"
    } else if has("unknown variable") {
        "E-name-unknown"
    } else if has("unknown function") {
        "E-call-unknown"
    } else if has("unknown type") || has("unknown record type") {
        "E-type-unknown"
    } else if has("has no field") || has("has no variant") {
        "E-field"
    } else if has("argument(s)") || (has(" expects ") && has("argument")) || (has(" takes ") && has(" argument")) {
        "E-arity"
    } else if has("keyword argument") || has("is given twice") {
        "E-keyword"
    } else if has("already defined") {
        "E-redefined"
    } else if has("`match`") || has("this arm can never run") {
        "E-match"
    } else {
        "E-check"
    }
}

/// A fix that is certain to be what the diagnostic says, as JSON
/// (`{"message","edits":[...]}`), for the hint rules. Columns are 1-based,
/// end exclusive, as the diagnostic's own.
pub fn fix_of(rule: &str, file: &str, line: usize, col: usize, end_col: usize, source: &str, notes: &[String]) -> Option<String> {
    let edit = |c0: usize, c1: usize, text: &str| {
        format!(r#"{{"file":{},"line":{line},"col":{c0},"end_line":{line},"end_col":{c1},"text":{}}}"#, json_str(file), json_str(text))
    };
    let fix = |msg: &str, e: String| format!(r#"{{"message":{},"edits":[{e}]}}"#, json_str(msg));
    match rule {
        "H-pow" => {
            // the `**` nearest the column (the parser points at its second `*`)
            let at = source.match_indices("**").map(|(i, _)| i + 1).min_by_key(|c| (*c as i64 - col as i64).abs())?;
            Some(fix("write `^` for the power", edit(at, at + 2, "^")))
        }
        "H-range" => {
            // `in range(n)` with one argument, as `to n`
            let i = source.find(" in range(")? + 1;
            let open = i + "in range(".len();
            let mut depth = 1;
            let close = source[open..].char_indices().find_map(|(k, ch)| {
                match ch {
                    '(' => depth += 1,
                    ')' => depth -= 1,
                    _ => {}
                }
                (depth == 0).then_some(open + k)
            })?;
            let n = source[open..close].trim();
            if n.is_empty() || n.contains(',') {
                return None;
            }
            Some(fix("loop with `for i to n`", edit(i + 1, close + 2, &format!("to {n}"))))
        }
        "H-juxtapose" => {
            let n = notes.iter().find(|n| n.starts_with("Viv has no implicit multiplication"))?;
            let want = n.split('`').nth(1)?;
            let have = source.get(col.checked_sub(1)?..end_col.checked_sub(1)?)?;
            if have.is_empty() || want.replace(" * ", "") != have {
                return None;
            }
            Some(fix(&format!("write `{want}`"), edit(col, end_col, want)))
        }
        _ => None,
    }
}

/// How much of a list or text a command prints: `--summary`, `--max-lines N`
/// (0: no bound), `--page K`.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub summary: bool,
    pub max_lines: usize,
    pub page: usize,
    pub given: bool,
}

pub const DEFAULT_MAX_LINES: usize = 200;
/// What `--summary` keeps of each list.
pub const SUMMARY_ITEMS: usize = 5;
/// What `--summary` keeps of a run's output: its last lines.
pub const SUMMARY_TAIL: usize = 20;

impl Default for Budget {
    fn default() -> Self {
        Budget { summary: false, max_lines: DEFAULT_MAX_LINES, page: 1, given: false }
    }
}

/// Take `--summary`, `--max-lines N` and `--page K` out of the arguments,
/// so the commands that parse their own see none of them.
pub fn take_budget(args: Vec<String>) -> Result<(Vec<String>, Budget), String> {
    let mut b = Budget::default();
    let mut out = Vec::with_capacity(args.len());
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--summary" => {
                b.summary = true;
                b.given = true;
            }
            "--max-lines" | "--page" => {
                let v = it.next().ok_or_else(|| format!("{a} takes a number"))?;
                let n: usize = v.parse().map_err(|_| format!("{a} takes a whole number, not `{v}`"))?;
                if a == "--page" {
                    b.page = n.max(1);
                } else {
                    b.max_lines = n;
                }
                b.given = true;
            }
            _ => out.push(a),
        }
    }
    Ok((out, b))
}

/// The one object a `--json` command prints.
pub fn envelope(cmd: &str, exit: u8, diagnostics: &str, result: &str, truncated: bool, more: Option<&str>) -> String {
    format!(
        r#"{{"viv":{SCHEMA},"cmd":{},"ok":{},"exit":{exit},"diagnostics":{diagnostics},"result":{result},"truncated":{truncated},"more":{}}}"#,
        json_str(cmd),
        exit == OK,
        more.map(json_str).unwrap_or_else(|| "null".into())
    )
}

/// `VIV_JSON_STDERR=1`: check and run write `{"diagnostics":[...]}` to
/// stderr as before the envelope (kept for one release).
pub fn legacy_stderr() -> bool {
    std::env::var("VIV_JSON_STDERR").map_or(false, |v| v == "1")
}

/// `VIV_JSON_RAW=1`: a wrapped command prints its own JSON, unwrapped (the
/// wrapper sets it on itself; a caller that wants the old object may too).
pub fn raw_json() -> bool {
    std::env::var("VIV_JSON_RAW").map_or(false, |v| v == "1")
}

/// Bound the lines of a text: the last `n` under `--summary`
/// (SUMMARY_TAIL), else page `page` of `max` lines. Returns the text, whether
/// it was cut, and the line count.
pub fn bound_text(text: &str, b: &Budget, tail: bool) -> (String, bool, usize) {
    let lines: Vec<&str> = text.lines().collect();
    let n = lines.len();
    let keep = if b.summary { SUMMARY_TAIL } else { b.max_lines };
    if keep == 0 || n <= keep {
        return (text.to_string(), false, n);
    }
    let part: Vec<&str> = if tail && b.page <= 1 {
        lines[n - keep..].to_vec()
    } else {
        let start = (b.page - 1) * keep;
        lines.iter().skip(start).take(keep).copied().collect()
    };
    (part.join("\n") + "\n", true, n)
}

/// Bound every list in an object (one level down): `--summary` keeps the
/// first SUMMARY_ITEMS and records each list's length in `counts`; otherwise
/// page `page` of `max_lines` items. Returns the object and whether a list
/// was cut.
pub fn bound_json(v: Json, b: &Budget) -> (Json, bool) {
    let keep = if b.summary { SUMMARY_ITEMS } else { b.max_lines };
    let Json::Obj(fields) = v else {
        return (v, false);
    };
    let mut cut = false;
    let mut counts: Vec<(String, Json)> = Vec::new();
    let mut out: Vec<(String, Json)> = Vec::new();
    for (k, x) in fields {
        match x {
            Json::Arr(items) => {
                counts.push((k.clone(), Json::Num(items.len().to_string())));
                if keep > 0 && items.len() > keep {
                    cut = true;
                    let start = if b.summary { 0 } else { (b.page - 1) * keep };
                    out.push((k, Json::Arr(items.into_iter().skip(start).take(keep).collect())));
                } else {
                    out.push((k, Json::Arr(items)));
                }
            }
            Json::Obj(inner) => {
                // one level further down, for objects such as {"defs": [...]}
                let (y, c) = bound_json(Json::Obj(inner), b);
                cut |= c;
                out.push((k, y));
            }
            other => out.push((k, other)),
        }
    }
    if b.summary && !counts.is_empty() {
        out.push(("counts".into(), Json::Obj(counts)));
    }
    (Json::Obj(out), cut)
}

/// The cursor a cut list or text prints: the same command with the next page.
pub fn more_cursor(b: &Budget) -> String {
    if b.summary {
        "drop --summary for every item, or pass --max-lines 0".into()
    } else {
        format!("--page {}", b.page + 1)
    }
}

// ---------------------------------------------------------------------------
// Wrapping a command's own JSON in the envelope

/// The commands whose `--json` this module wraps.
pub const WRAPPED: &[&str] = &["features", "explain", "claims"];

/// Run viv again with `args` and VIV_JSON_RAW=1; (exit code, stdout, stderr).
fn rerun(args: &[String], extra_env: &[(&str, &str)]) -> Result<(i32, String, String), String> {
    let me = std::env::current_exe().map_err(|e| format!("cannot find the viv binary: {e}"))?;
    let mut c = Command::new(me);
    c.args(args).env("VIV_JSON_RAW", "1").stdin(Stdio::null());
    for (k, v) in extra_env {
        c.env(k, v);
    }
    let o = c.output().map_err(|e| format!("cannot run viv: {e}"))?;
    Ok((o.status.code().unwrap_or(INTERNAL as i32), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned()))
}

/// A plain diagnostic (no file) as JSON, with its rule.
pub fn plain_diag(message: &str, rule: &str) -> String {
    format!(
        r#"{{"severity":"error","rule":{},"message":{},"file":null,"line":null,"col":null,"notes":[]}}"#,
        json_str(rule),
        json_str(message.strip_prefix("error: ").unwrap_or(message))
    )
}

/// The rule for a failed wrapped command, from its stderr.
fn failure_rule(cmd: &str, err: &str, code: i32) -> &'static str {
    let e = err.to_lowercase();
    if e.contains("usage") || e.contains("unknown argument") || e.contains("cannot read") || e.contains("no such file") || e.contains("no def `") || e.contains("not found") {
        if e.contains("cannot read") || e.contains("no such file") {
            "U-file"
        } else {
            "U-usage"
        }
    } else if cmd == "claims" && code == 1 {
        "T-fail"
    } else if code == INTERNAL as i32 {
        "I-internal"
    } else {
        "E-check"
    }
}

/// `viv features|explain|claims ... --json`: the command's own JSON as the
/// envelope's `result`.
pub fn wrap(cmd: &str, args: &[String], b: &Budget) -> ExitCode {
    let mut child: Vec<String> = args[1..].to_vec();
    // `--summary` on explain: the top two levels
    if cmd == "explain" && b.summary && !child.iter().any(|a| a == "--depth") {
        child.push("--depth".into());
        child.push("2".into());
    }
    // features: only `--json` alone and `--usage --json` print JSON; a
    // search or one entry prints text, read here into a list
    let search = cmd == "features" && child.iter().any(|a| a == "--search");
    let one = cmd == "features" && !search && child.iter().skip(1).any(|a| !a.starts_with("--"));
    if search || one {
        child.retain(|a| a != "--json");
    }
    let (code, out, err) = match rerun(&child, &[]) {
        Ok(x) => x,
        Err(e) => return print_envelope(cmd, INTERNAL, &format!("[{}]", plain_diag(&e, "I-internal")), "null", false, None),
    };
    if code != 0 {
        let rule = failure_rule(cmd, &err, code);
        let msg = err.trim();
        let msg = if msg.is_empty() { out.trim() } else { msg };
        let first: String = msg.lines().take(SUMMARY_TAIL).collect::<Vec<_>>().join("\n");
        let result = if cmd == "claims" { parse_json(out.trim()).map(|j| j.text()).unwrap_or_else(|_| "null".into()) } else { "null".into() };
        return print_envelope(cmd, exit_for_rule(rule), &format!("[{}]", plain_diag(&first, rule)), &result, false, None);
    }
    let value = if search {
        Json::Obj(vec![("entries".into(), Json::Arr(out.lines().filter_map(search_row).collect()))])
    } else if one {
        Json::Obj(vec![("name".into(), Json::Str(child.iter().skip(1).find(|a| !a.starts_with("--")).cloned().unwrap_or_default())), ("text".into(), Json::Str(out.clone()))])
    } else {
        match parse_json(out.trim()) {
            Ok(j) => j,
            Err(_) => Json::Obj(vec![("text".into(), Json::Str(out.clone()))]),
        }
    };
    let (value, cut) = bound_json(value, b);
    let more = cut.then(|| more_cursor(b));
    print_envelope(cmd, OK, "[]", &value.text(), cut, more.as_deref())
}

/// One line of `viv features --search` (`  name   area   text`) as an object.
fn search_row(line: &str) -> Option<Json> {
    let t = line.trim();
    if t.is_empty() {
        return None;
    }
    let cols: Vec<&str> = t.split("  ").map(str::trim).filter(|c| !c.is_empty()).collect();
    if cols.len() < 3 {
        return None;
    }
    Some(Json::Obj(vec![
        ("name".into(), Json::Str(cols[0].into())),
        ("area".into(), Json::Str(cols[1].into())),
        ("what".into(), Json::Str(cols[2..].join(" "))),
    ]))
}

pub fn print_envelope(cmd: &str, exit: u8, diags: &str, result: &str, truncated: bool, more: Option<&str>) -> ExitCode {
    println!("{}", envelope(cmd, exit, diags, result, truncated, more));
    ExitCode::from(exit)
}

// ---------------------------------------------------------------------------
// viv map

const MAP_USAGE: &str = "usage: viv map <file.viv> [<def>] [--text | --json] [--summary] [--max-lines N]";

/// `viv map <file> [<def>] [--text|--json]`: the code map (codemap.rs) as
/// indented text, one def per line, or as the envelope.
pub fn map(args: &[String], b: &Budget) -> ExitCode {
    let json = args.iter().any(|a| a == "--json");
    let pos: Vec<&String> = args[2..].iter().filter(|a| !a.starts_with("--")).collect();
    let fail = |code: u8, rule: &str, msg: &str| -> ExitCode {
        if json {
            print_envelope("map", code, &format!("[{}]", plain_diag(msg, rule)), "null", false, None)
        } else {
            eprintln!("{msg}");
            ExitCode::from(code)
        }
    };
    let Some(file) = pos.first() else {
        return fail(USAGE, "U-usage", MAP_USAGE);
    };
    if !Path::new(file.as_str()).exists() {
        return fail(USAGE, "U-file", &format!("error: cannot read {file}: no such file"));
    }
    let mut child: Vec<String> = vec!["graph".into(), (*file).clone(), "--map".into()];
    if let Some(d) = pos.get(1) {
        child.push("--entry".into());
        child.push((*d).clone());
    }
    let (code, out, err) = match rerun(&child, &[]) {
        Ok(x) => x,
        Err(e) => return fail(INTERNAL, "I-internal", &e),
    };
    if code != 0 {
        let rule = if err.contains("no def `") { "U-usage" } else { "E-check" };
        return fail(exit_for_rule(rule), rule, err.trim());
    }
    let Ok(m) = parse_json(out.trim()) else {
        return fail(INTERNAL, "I-internal", "viv graph --map did not print JSON");
    };
    if json {
        // the modules' full sources stay out: `viv src` gives one def
        let m = strip_sources(m);
        let (m, cut) = bound_json(m, b);
        let more = cut.then(|| more_cursor(b));
        return print_envelope("map", OK, "[]", &m.text(), cut, more.as_deref());
    }
    let text = map_text(&m);
    let (text, cut, n) = bound_text(&text, b, false);
    print!("{text}");
    if cut {
        println!("... {} more lines: {}", n.saturating_sub(text.lines().count()), more_cursor(b));
    }
    ExitCode::SUCCESS
}

/// How much of an intent a line of `viv map --text` shows.
const INTENT_CHARS: usize = 90;

/// The first sentence of a text, at most INTENT_CHARS characters.
fn first_sentence(t: &str) -> String {
    let end = t.find(". ").map(|i| i + 1).unwrap_or(t.len());
    let s = &t[..end];
    if s.chars().count() > INTENT_CHARS {
        s.chars().take(INTENT_CHARS - 3).collect::<String>() + "..."
    } else {
        s.to_string()
    }
}

fn strip_sources(m: Json) -> Json {
    let Json::Obj(fs) = m else { return m };
    Json::Obj(
        fs.into_iter()
            .map(|(k, v)| {
                if k == "modules" {
                    if let Json::Arr(ms) = v {
                        let ms = ms.into_iter().map(|x| if let Json::Obj(f) = x { Json::Obj(f.into_iter().filter(|(k, _)| k != "src").collect()) } else { x }).collect();
                        return (k, Json::Arr(ms));
                    }
                    (k, v)
                } else {
                    (k, v)
                }
            })
            .collect(),
    )
}

/// The code map as text: the root and its path, then each def under the def
/// it is first reached from, with its intent, its loops and, for a contract,
/// its methods.
pub fn map_text(m: &Json) -> String {
    let arr = |v: Option<&Json>| -> Vec<Json> {
        if let Some(Json::Arr(a)) = v { a.clone() } else { Vec::new() }
    };
    let s = |v: Option<&Json>| v.and_then(Json::str).unwrap_or("").to_string();
    let modules = arr(m.get("modules"));
    let defs = arr(m.get("defs"));
    let contracts = arr(m.get("contracts"));
    let root_path = modules.first().map(|x| s(x.get("path"))).unwrap_or_default();
    let mut out = format!("{} ({})\n", s(m.get("root")), root_path);
    let ids: Vec<String> = defs.iter().map(|d| s(d.get("id"))).collect();
    let from: Vec<Option<String>> = defs.iter().map(|d| d.get("from").and_then(Json::str).map(str::to_string)).collect();
    let width = ids.iter().map(|i| i.len()).max().unwrap_or(0);
    // depth-first from the entry, children in the order the map reached them
    fn walk(k: usize, depth: usize, ids: &[String], from: &[Option<String>], line: &dyn Fn(usize, usize) -> String, out: &mut String) {
        out.push_str(&line(k, depth));
        for c in 0..ids.len() {
            if from[c].as_deref() == Some(ids[k].as_str()) {
                walk(c, depth + 1, ids, from, line, out);
            }
        }
    }
    let line = |k: usize, depth: usize| -> String {
        let d = &defs[k];
        let id = &ids[k];
        let mut what = s(d.get("intent"));
        if what.is_empty() {
            what = s(d.get("doc"));
        }
        let what = first_sentence(&what.split_whitespace().collect::<Vec<_>>().join(" "));
        let mut extra = String::new();
        if let Some(c) = contracts.iter().find(|c| s(c.get("name")) == *id) {
            let ms: Vec<String> = arr(c.get("methods")).iter().filter_map(|x| x.str().map(str::to_string)).collect();
            extra.push_str(&format!("   contract: {}", ms.join(" | ")));
        }
        let loops: Vec<String> = arr(d.get("loops")).iter().filter(|l| l.get("depth").and_then(Json::num) == Some(0.0)).map(|l| s(l.get("head"))).collect();
        if !loops.is_empty() {
            let shown: Vec<String> = loops.iter().take(2).map(|h| h.trim().trim_end_matches('{').trim().to_string()).collect();
            extra.push_str(&format!("   [loop: {}{}]", shown.join("; "), if loops.len() > 2 { "; ..." } else { "" }));
        }
        let at = d.get("line").and_then(Json::num).unwrap_or(0.0) as usize;
        let module = d.get("module").and_then(Json::num).unwrap_or(0.0) as usize;
        let place = if id == "<main>" { String::new() } else { format!("  {}:{at}", modules.get(module).map(|x| s(x.get("name"))).unwrap_or_default()) };
        format!("{}{:<width$}  {what}{extra}{place}\n", "  ".repeat(depth + 1), id, width = width)
    };
    if !ids.is_empty() {
        walk(0, 0, &ids, &from, &line, &mut out);
    }
    for c in &contracts {
        if !ids.contains(&s(c.get("name"))) {
            let ms: Vec<String> = arr(c.get("methods")).iter().filter_map(|x| x.str().map(str::to_string)).collect();
            out.push_str(&format!("  contract {}: {}   (using {})\n", s(c.get("name")), ms.join(" | "), s(c.get("selector"))));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// viv help

/// One command for `viv help`: name, usage, what it does, and whether its
/// `--json` follows the envelope.
pub struct Cmd {
    pub name: &'static str,
    pub group: &'static str,
    pub usage: &'static str,
    pub what: &'static str,
    pub schema: Option<&'static str>,
}

pub const COMMANDS: &[Cmd] = &[
    Cmd { name: "help", group: "learn", usage: "viv help [--agent [--brief] | <command> | rules | exit-codes | json] [--json]", what: "this list; --agent: the working guide for an agent", schema: Some(include_str!("../schemas/help.json")) },
    Cmd { name: "features", group: "learn", usage: "viv features [<name> | --search <words> | --usage [<path>...]] [--json] [--summary]", what: "the catalog of what Viv can do: consult it before writing code", schema: Some(include_str!("../schemas/features.json")) },
    Cmd { name: "explain", group: "learn", usage: "viv explain <file.viv> [--from <def>] [--depth <k>] [--json] [--summary]", what: "a program's intent from the top down", schema: Some(include_str!("../schemas/explain.json")) },
    Cmd { name: "map", group: "learn", usage: "viv map <file.viv> [<def>] [--text | --json] [--summary] [--max-lines N]", what: "the code map: each def under its caller, with intent, loops and contracts", schema: Some(include_str!("../schemas/map.json")) },
    Cmd { name: "atlas", group: "learn", usage: "viv atlas <file.viv> [-o <file.atlas.html>] [--text | --json] [--no-run] [--no-walk]", what: "the atlas: every def by module, the calls between them and one run's counts, with a walk through that run line by line, as a page that opens offline", schema: None },
    Cmd { name: "whatis", group: "learn", usage: "viv whatis <file.viv> <name>", what: "what one name in a program is", schema: None },
    Cmd { name: "lsfunc", group: "learn", usage: "viv lsfunc <file.viv>", what: "the table of contents: every function and its doc", schema: None },
    Cmd { name: "src", group: "learn", usage: "viv src <file.viv> <def>", what: "one def's source", schema: None },
    Cmd { name: "check", group: "write", usage: "viv check <file.viv | file.py> [--set NAME=value] [--using C=m] [--json] [--summary] [--max-lines N]", what: "parse and type-check without running", schema: Some(include_str!("../schemas/check.json")) },
    Cmd { name: "fix", group: "write", usage: "viv fix --keywords [--write] [--json] <file.viv>... | viv fix --apply-json <envelope.json>", what: "name positional options; apply the edits a --json diagnostic carries", schema: Some(include_str!("../schemas/fix.json")) },
    Cmd { name: "lint", group: "write", usage: "viv lint <file.viv>", what: "size and style rules", schema: None },
    Cmd { name: "run", group: "run", usage: "viv run <file.viv | file.py> [--set NAME=value] [--using C=m] [--quiet] [--json] [--summary] [--max-lines N]", what: "run the program and its tests in the interpreter", schema: Some(include_str!("../schemas/run.json")) },
    Cmd { name: "digits", group: "run", usage: "viv digits <file.viv> [--json]", what: "which printed digits survive perturbed arithmetic", schema: None },
    Cmd { name: "claims", group: "record", usage: "viv claims [--json] [--only <slug>] | viv claims run [--all | --heavy | --only <slug>] [--write] [--check]", what: "claims (a bounded statement, its program and its vipyr twin) and their status", schema: Some(include_str!("../schemas/claims.json")) },
    Cmd { name: "build", group: "ship", usage: "viv build <file.viv> ...", what: "a native program with the C compiler", schema: None },
    Cmd { name: "export", group: "ship", usage: "viv export <manifest.toml>", what: "kernels and libraries for Python", schema: None },
    Cmd { name: "skill", group: "agent", usage: "viv skill install [--project | --user | --plugin] [--force] [--json]", what: "write the vipyr skill for Claude Code into .claude/skills", schema: Some(include_str!("../schemas/skill.json")) },
    Cmd { name: "hook", group: "agent", usage: "viv hook post-edit   (reads a Claude Code PostToolUse event on stdin)", what: "check an edited .viv file; exit 2 with the errors on stderr", schema: None },
];

/// The vipyr side, listed by `viv help` so an agent sees both.
const VIPYR_COMMANDS: &[(&str, &str)] = &[
    ("vipyr catalog [--topic T] [--json]", "the validated engines: what each solves, its reference, claims, code map, and the timings measured beside plain Python where they exist"),
    ("vipyr help [--agent [--brief]]", "the front door's help; --agent leads with the catalog"),
    ("vipyr run|check|features|explain|map|claims ...", "the same commands as viv, through the bundled compiler"),
    ("vipyr skill install", "the same as viv skill install"),
    ("vipyr stubs|build|compile|compiler", "Python stubs, compiled modules, scikit-learn to C, the compiler's path"),
];

pub const AGENT_GUIDE: &str = include_str!("../agent/guide.md");
pub const ENGINES: &str = include_str!("../../../python/vipyr/engines.json");

fn find_cmd(name: &str) -> Option<&'static Cmd> {
    COMMANDS.iter().find(|c| c.name == name)
}

/// `viv help ...`. `json` asks for the envelope.
pub fn help(args: &[String]) -> ExitCode {
    let rest: Vec<&str> = args.iter().skip(2).map(String::as_str).filter(|a| *a != "--json").collect();
    let json = args.iter().any(|a| a == "--json");
    match rest.first().copied() {
        None => {
            if json {
                let cmds: Vec<String> = COMMANDS.iter().map(|c| format!(r#"{{"name":{},"group":{},"usage":{},"what":{}}}"#, json_str(c.name), json_str(c.group), json_str(c.usage), json_str(c.what))).collect();
                return print_envelope("help", OK, "[]", &format!(r#"{{"commands":[{}]}}"#, cmds.join(",")), false, None);
            }
            print!("{}", overview());
            ExitCode::SUCCESS
        }
        Some("--agent") => {
            let brief = rest.contains(&"--brief");
            let text = agent_guide(brief);
            if json {
                return print_envelope("help", OK, "[]", &format!(r#"{{"text":{}}}"#, json_str(&text)), false, None);
            }
            print!("{text}");
            ExitCode::SUCCESS
        }
        Some("rules") => {
            if json {
                let rs: Vec<String> = RULES.iter().map(|(id, what, ex)| format!(r#"{{"id":{},"exit":{},"what":{},"example":{}}}"#, json_str(id), exit_for_rule(id), json_str(what), json_str(ex))).collect();
                return print_envelope("help", OK, "[]", &format!(r#"{{"rules":[{}]}}"#, rs.join(",")), false, None);
            }
            for (id, what, ex) in RULES {
                println!("{id:<15} exit {:<3} {what}{}", exit_for_rule(id), if ex.is_empty() { String::new() } else { format!("   e.g. {ex}") });
            }
            ExitCode::SUCCESS
        }
        Some("exit-codes") => {
            if json {
                let cs: Vec<String> = EXIT_CODES.iter().map(|(c, n, w)| format!(r#"{{"code":{c},"name":{},"what":{}}}"#, json_str(n), json_str(w))).collect();
                return print_envelope("help", OK, "[]", &format!(r#"{{"exit_codes":[{}]}}"#, cs.join(",")), false, None);
            }
            for (c, n, w) in EXIT_CODES {
                println!("{c:<3} {n:<12} {w}");
            }
            ExitCode::SUCCESS
        }
        Some("json") => {
            println!("{}", JSON_HELP.trim());
            ExitCode::SUCCESS
        }
        Some(name) => match find_cmd(name) {
            Some(c) => {
                if json {
                    let schema = c.schema.map(|s| s.trim().to_string()).unwrap_or_else(|| "null".into());
                    return print_envelope("help", OK, "[]", &format!(r#"{{"name":{},"usage":{},"what":{},"envelope":{},"schema":{schema}}}"#, json_str(c.name), json_str(c.usage), json_str(c.what), c.schema.is_some()), false, None);
                }
                println!("{}\n\n  {}", c.usage, c.what);
                if c.schema.is_some() {
                    println!("\n  --json prints the envelope (viv help json); the result's schema: viv help {} --json", c.name);
                }
                ExitCode::SUCCESS
            }
            None => {
                let msg = format!("viv help: no command `{name}` (viv help lists them)");
                if json {
                    return print_envelope("help", USAGE, &format!("[{}]", plain_diag(&msg, "U-usage")), "null", false, None);
                }
                eprintln!("{msg}");
                ExitCode::from(USAGE)
            }
        },
    }
}

const JSON_HELP: &str = r#"
With --json, check, run, features, explain, map, claims, help, skill and
fix print one object on stdout and nothing else there:

  {"viv": 1, "cmd": "check", "ok": false, "exit": 3,
   "diagnostics": [{"severity", "rule", "message", "file", "line", "col",
                    "end_line", "end_col", "label", "source", "notes",
                    "fix": {"message", "edits": [{"file", "line", "col",
                            "end_line", "end_col", "text"}]}}],
   "result": {...}, "truncated": false, "more": null}

  viv       the schema version
  ok        exit == 0
  exit      the process's exit code (viv help exit-codes)
  rule      a stable id (viv help rules)
  fix       present only when the edit is certain; viv fix --apply-json
            applies it
  result    the command's own payload (viv help <command> --json gives
            its schema)
  truncated a list or text was cut: --summary keeps counts and the first
            5 items (a run: its last 20 lines), --max-lines N (default 200,
            0 for no bound) bounds each list, and `more` names the flag for
            the next part

VIV_JSON_STDERR=1 gives check and run their old form for one release:
{"diagnostics": [...]} on stderr.
"#;

fn overview() -> String {
    let mut s = String::from(
        "viv: the Viv language's tool. vipyr is the front door: the same commands\n\
         plus the catalog of validated engines; one app, sharing runs and claims.\n\
         vipyr is not on PyPI yet; from a checkout, use PYTHONPATH=python.\n\n",
    );
    for g in ["learn", "write", "run", "record", "ship", "agent"] {
        s.push_str(&format!("{g}\n"));
        for c in COMMANDS.iter().filter(|c| c.group == g) {
            s.push_str(&format!("  {:<9} {}\n", c.name, c.what));
        }
    }
    s.push_str("vipyr\n");
    for (u, w) in VIPYR_COMMANDS {
        s.push_str(&format!("  {u}\n      {w}\n"));
    }
    s.push_str(
        "\nviv help <command>     usage and flags          viv help --agent   the guide for an agent\n\
         viv help rules         diagnostic rule ids      viv help exit-codes\n\
         viv help json          the --json envelope\n",
    );
    s
}

/// The working guide (agent/guide.md) with the catalog in its place. `brief`: the
/// first 30 lines, for a session-start hook.
pub fn agent_guide(brief: bool) -> String {
    let s = AGENT_GUIDE.replace("{{catalog}}", &catalog_brief());
    if brief {
        let lines: Vec<&str> = s.lines().take(BRIEF_LINES).collect();
        return lines.join("\n") + "\n... (viv help --agent for the rest)\n";
    }
    s
}

/// How much of an engine's `solves` the guide shows.
const SOLVES_CHARS: usize = 72;

/// How many lines `--brief` keeps.
pub const BRIEF_LINES: usize = 30;

/// The catalog in a few lines: each engine's id, what it solves and, where
/// one was measured, the comparison with plain Python.
pub fn catalog_brief() -> String {
    let Ok(j) = parse_json(ENGINES) else {
        return "Validated engines: vipyr catalog --json\n".into();
    };
    let engines = match j.get("engines") {
        Some(Json::Arr(a)) => a.clone(),
        _ => Vec::new(),
    };
    let mut s = format!(
        "Validated engines ({}; `vipyr catalog --json` gives each one's reference, agreement,\nclaims, code map and the timings measured beside plain Python). Use one before writing a solver:\n",
        engines.len()
    );
    for e in &engines {
        let g = |k: &str| e.get(k).and_then(Json::str).unwrap_or("").to_string();
        let solves: String = g("solves").chars().take(SOLVES_CHARS).collect();
        s.push_str(&format!("  {:<22} {solves}\n", g("id")));
        if let Some(c) = e.get("comparison").filter(|c| **c != Json::Null) {
            let against = c.get("against").and_then(Json::str).unwrap_or("");
            if let Some(x) = c.get("speed").and_then(Json::str) {
                s.push_str(&format!("  {:<22}   measured with {against}: {x}\n", ""));
            }
        }
    }
    s
}

// ---------------------------------------------------------------------------
// viv skill install, viv hook post-edit

/// The skill's files, as `viv skill install` writes them (relative to the
/// skill's directory).
pub const SKILL_FILES: &[(&str, &str)] = &[
    ("SKILL.md", include_str!("../../../claude/vipyr/skills/vipyr/SKILL.md")),
    ("reference/house.md", include_str!("../../../claude/vipyr/skills/vipyr/reference/house.md")),
    ("reference/claims.md", include_str!("../../../claude/vipyr/skills/vipyr/reference/claims.md")),
];

/// The allowlist the skill prints for the user to add (phase 1: read-only
/// commands only; nothing here runs a program).
pub const ALLOWLIST: &[&str] = &[
    "Bash(viv help *)", "Bash(viv features *)", "Bash(viv explain *)", "Bash(viv map *)", "Bash(viv whatis *)",
    "Bash(viv src *)", "Bash(viv lsfunc *)", "Bash(viv check *)", "Bash(viv claims)", "Bash(viv claims --json*)",
    "Bash(vipyr help *)", "Bash(vipyr catalog *)",
];

/// A 64-bit FNV-1a hash, as hex: enough to tell whether a file is one viv wrote.
pub fn fnv(text: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in text.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

/// The file beside an installed skill that lists the hashes of every
/// SKILL.md viv has written there.
const STAMP: &str = ".vipyr-installed";

/// `viv skill install [--project|--user|--plugin] [--force] [--json]`.
pub fn skill(args: &[String]) -> ExitCode {
    let json = args.iter().any(|a| a == "--json");
    let say = |code: u8, rule: Option<&str>, msg: &str, result: &str| -> ExitCode {
        if json {
            let d = rule.map(|r| format!("[{}]", plain_diag(msg, r))).unwrap_or_else(|| "[]".into());
            return print_envelope("skill", code, &d, result, false, None);
        }
        if code == OK {
            print!("{msg}");
        } else {
            eprintln!("{msg}");
        }
        ExitCode::from(code)
    };
    if args.get(2).map(String::as_str) != Some("install") {
        return say(USAGE, Some("U-usage"), "usage: viv skill install [--project | --user | --plugin] [--force] [--json]", "null");
    }
    let flag = |f: &str| args.iter().any(|a| a == f);
    if flag("--plugin") {
        let msg = "Claude Code installs plugins itself. From a checkout of the Viv repository:\n  claude plugin marketplace add ./claude\n  claude plugin install vipyr@vipyr\n";
        return say(OK, None, msg, &format!(r#"{{"commands":[{},{}]}}"#, json_str("claude plugin marketplace add ./claude"), json_str("claude plugin install vipyr@vipyr")));
    }
    let base: PathBuf = if flag("--user") {
        match std::env::var_os("HOME") {
            Some(h) => PathBuf::from(h).join(".claude/skills/vipyr"),
            None => return say(USAGE, Some("U-usage"), "viv skill install --user: HOME is not set", "null"),
        }
    } else {
        PathBuf::from(".claude/skills/vipyr")
    };
    let skill_path = base.join("SKILL.md");
    let stamp_path = base.join(STAMP);
    let known: Vec<String> = std::fs::read_to_string(&stamp_path).map(|t| t.lines().map(str::to_string).collect()).unwrap_or_default();
    if let Ok(now) = std::fs::read_to_string(&skill_path) {
        let h = fnv(&now);
        if h != fnv(SKILL_FILES[0].1) && !known.contains(&h) && !flag("--force") {
            let msg = format!("viv skill install: {} was edited since viv wrote it; --force replaces it", skill_path.display());
            return say(POLICY, Some("X-policy-overwrite"), &msg, "null");
        }
    }
    let mut written = Vec::new();
    for (rel, text) in SKILL_FILES {
        let p = base.join(rel);
        if let Some(d) = p.parent() {
            if let Err(e) = std::fs::create_dir_all(d) {
                return say(USAGE, Some("U-file"), &format!("viv skill install: cannot create {}: {e}", d.display()), "null");
            }
        }
        if let Err(e) = std::fs::write(&p, text) {
            return say(USAGE, Some("U-file"), &format!("viv skill install: cannot write {}: {e}", p.display()), "null");
        }
        written.push(p.display().to_string());
    }
    let mut stamps = known;
    let h = fnv(SKILL_FILES[0].1);
    if !stamps.contains(&h) {
        stamps.push(h);
    }
    let _ = std::fs::write(&stamp_path, stamps.join("\n") + "\n");
    let allow = ALLOWLIST.iter().map(|a| format!("      \"{a}\"")).collect::<Vec<_>>().join(",\n");
    let msg = format!(
        "wrote {}\n\nviv does not edit settings. To run the read-only commands without a prompt, add to .claude/settings.json:\n  {{\n    \"permissions\": {{\n      \"allow\": [\n{allow}\n      ]\n    }}\n  }}\n",
        written.join(", ")
    );
    let result = format!(
        r#"{{"dir":{},"files":[{}],"allow":[{}]}}"#,
        json_str(&base.display().to_string()),
        written.iter().map(|w| json_str(w)).collect::<Vec<_>>().join(","),
        ALLOWLIST.iter().map(|a| json_str(a)).collect::<Vec<_>>().join(",")
    );
    say(OK, None, &msg, &result)
}

/// The hook's time budget: past it, the hook lets the edit stand silently.
const HOOK_SECONDS: u64 = 10;
/// The most stderr lines the hook shows Claude.
const HOOK_LINES: usize = 15;

/// `viv hook post-edit`: read a PostToolUse event on stdin; for a .viv file,
/// run `viv check --json` on it and, when it has errors, print them (rule,
/// file:line:col, message, fix) to stderr and exit 2, which Claude Code shows
/// to the model. Anything else, or a check past HOOK_SECONDS, exits 0.
pub fn hook(args: &[String]) -> ExitCode {
    if args.get(2).map(String::as_str) != Some("post-edit") {
        eprintln!("usage: viv hook post-edit   (reads a Claude Code PostToolUse event on stdin)");
        return ExitCode::from(USAGE);
    }
    let mut input = String::new();
    if std::io::stdin().read_to_string(&mut input).is_err() {
        return ExitCode::SUCCESS;
    }
    let Ok(ev) = parse_json(input.trim()) else {
        return ExitCode::SUCCESS;
    };
    let Some(file) = ev.get("tool_input").and_then(|t| t.get("file_path")).and_then(Json::str).map(str::to_string) else {
        return ExitCode::SUCCESS;
    };
    if !file.ends_with(".viv") || !Path::new(&file).exists() {
        return ExitCode::SUCCESS;
    }
    let Ok(me) = std::env::current_exe() else {
        return ExitCode::SUCCESS;
    };
    let cwd = ev.get("cwd").and_then(Json::str).map(PathBuf::from);
    let mut c = Command::new(me);
    c.args(["check", "--json", file.as_str()]).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
    if let Some(d) = cwd.filter(|d| d.is_dir()) {
        c.current_dir(d);
    }
    let Ok(mut child) = c.spawn() else {
        return ExitCode::SUCCESS;
    };
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if start.elapsed() < Duration::from_secs(HOOK_SECONDS) => std::thread::sleep(Duration::from_millis(20)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return ExitCode::SUCCESS;
            }
        }
    }
    let mut out = String::new();
    if let Some(mut so) = child.stdout.take() {
        let _ = so.read_to_string(&mut out);
    }
    let Ok(env) = parse_json(out.trim()) else {
        return ExitCode::SUCCESS;
    };
    if env.get("ok") == Some(&Json::Bool(true)) {
        return ExitCode::SUCCESS;
    }
    let diags = match env.get("diagnostics") {
        Some(Json::Arr(d)) => d.clone(),
        _ => Vec::new(),
    };
    let mut lines: Vec<String> = Vec::new();
    for d in &diags {
        let g = |k: &str| d.get(k).and_then(Json::str).unwrap_or("").to_string();
        let n = |k: &str| d.get(k).and_then(Json::num).map(|x| (x as usize).to_string()).unwrap_or_else(|| "?".into());
        lines.push(format!("{}:{}:{}: {} {}", g("file"), n("line"), n("col"), g("rule"), g("message")));
        if let Some(Json::Arr(notes)) = d.get("notes") {
            for x in notes.iter().filter_map(Json::str) {
                lines.push(format!("  note: {x}"));
            }
        }
        if let Some(f) = d.get("fix") {
            let edits = match f.get("edits") {
                Some(Json::Arr(e)) => e.iter().map(|e| format!("line {} col {}..{} -> {}", e.get("line").map(Json::text).unwrap_or_default(), e.get("col").map(Json::text).unwrap_or_default(), e.get("end_col").map(Json::text).unwrap_or_default(), e.get("text").map(Json::text).unwrap_or_default())).collect::<Vec<_>>().join("; "),
                _ => String::new(),
            };
            lines.push(format!("  fix: {} ({edits})", f.get("message").and_then(Json::str).unwrap_or("")));
        }
    }
    if lines.is_empty() {
        return ExitCode::SUCCESS;
    }
    let total = lines.len();
    let mut err = std::io::stderr();
    let _ = writeln!(err, "viv check {file}:");
    for l in lines.iter().take(HOOK_LINES) {
        let _ = writeln!(err, "{l}");
    }
    if total > HOOK_LINES {
        let _ = writeln!(err, "... {} more: viv check {file} --json", total - HOOK_LINES);
    }
    ExitCode::from(USAGE)
}

// ---------------------------------------------------------------------------
// viv fix --apply-json

/// `viv fix --apply-json <file.json | ->`: apply every `fix.edits` in a
/// `--json` envelope (or a bare `{"diagnostics": [...]}`) to the files they
/// name. Edits in one file are applied from the last to the first so the
/// earlier columns stay right; overlapping edits are refused.
pub fn apply_json(args: &[String]) -> ExitCode {
    let json_out = args.iter().any(|a| a == "--json");
    let src_arg = args.iter().skip_while(|a| *a != "--apply-json").nth(1).cloned();
    let fail = |code: u8, rule: &str, msg: &str| -> ExitCode {
        if json_out {
            return print_envelope("fix", code, &format!("[{}]", plain_diag(msg, rule)), "null", false, None);
        }
        eprintln!("{msg}");
        ExitCode::from(code)
    };
    let Some(src_arg) = src_arg else {
        return fail(USAGE, "U-usage", "usage: viv fix --apply-json <envelope.json | ->");
    };
    let text = if src_arg == "-" {
        let mut s = String::new();
        let _ = std::io::stdin().read_to_string(&mut s);
        s
    } else {
        match std::fs::read_to_string(&src_arg) {
            Ok(s) => s,
            Err(e) => return fail(USAGE, "U-file", &format!("error: cannot read {src_arg}: {e}")),
        }
    };
    let Ok(j) = parse_json(text.trim()) else {
        return fail(USAGE, "U-usage", "viv fix --apply-json: the input is not JSON");
    };
    let diags = match j.get("diagnostics") {
        Some(Json::Arr(d)) => d.clone(),
        _ => Vec::new(),
    };
    // file -> [(line, col, end_line, end_col, text)]
    let mut by_file: Vec<(String, Vec<(usize, usize, usize, usize, String)>)> = Vec::new();
    for d in &diags {
        let Some(Json::Arr(edits)) = d.get("fix").and_then(|f| f.get("edits")) else { continue };
        for e in edits {
            let n = |k: &str| e.get(k).and_then(Json::num).unwrap_or(0.0) as usize;
            let file = e.get("file").and_then(Json::str).unwrap_or("").to_string();
            let t = e.get("text").and_then(Json::str).unwrap_or("").to_string();
            let entry = match by_file.iter_mut().find(|(f, _)| *f == file) {
                Some(x) => x,
                None => {
                    by_file.push((file.clone(), Vec::new()));
                    by_file.last_mut().unwrap()
                }
            };
            entry.1.push((n("line"), n("col"), n("end_line"), n("end_col"), t));
        }
    }
    let mut applied = 0usize;
    let mut files = Vec::new();
    for (file, mut edits) in by_file {
        let Ok(src) = std::fs::read_to_string(&file) else {
            return fail(USAGE, "U-file", &format!("error: cannot read {file}"));
        };
        let starts: Vec<usize> = std::iter::once(0).chain(src.match_indices('\n').map(|(i, _)| i + 1)).collect();
        let off = |l: usize, c: usize| starts.get(l.wrapping_sub(1)).map(|s| s + c.saturating_sub(1));
        edits.sort_by(|a, b| (b.0, b.1).cmp(&(a.0, a.1)));
        let mut out = src.clone();
        let mut last_start = usize::MAX;
        for (l, c, el, ec, t) in &edits {
            let (Some(a), Some(z)) = (off(*l, *c), off(*el, *ec)) else {
                return fail(USAGE, "U-usage", &format!("viv fix --apply-json: an edit outside {file}"));
            };
            if z > last_start || a > z || z > out.len() {
                return fail(USAGE, "U-usage", &format!("viv fix --apply-json: overlapping edits in {file}"));
            }
            out.replace_range(a..z, t);
            last_start = a;
            applied += 1;
        }
        if let Err(e) = std::fs::write(&file, out) {
            return fail(USAGE, "U-file", &format!("error: cannot write {file}: {e}"));
        }
        files.push(file);
    }
    if json_out {
        let r = format!(r#"{{"applied":{applied},"files":[{}]}}"#, files.iter().map(|f| json_str(f)).collect::<Vec<_>>().join(","));
        return print_envelope("fix", OK, "[]", &r, false, None);
    }
    println!("{applied} edit{} applied in {} file{}", if applied == 1 { "" } else { "s" }, files.len(), if files.len() == 1 { "" } else { "s" });
    ExitCode::SUCCESS
}

/// Line-level edits that turn `old` into `new` when both have the same lines
/// (viv fix --keywords only inserts `name=` inside lines), else one edit of
/// the whole file.
pub fn line_edits(file: &str, old: &str, new: &str) -> Vec<String> {
    let a: Vec<&str> = old.split('\n').collect();
    let b: Vec<&str> = new.split('\n').collect();
    let edit = |l: usize, el: usize, ec: usize, t: &str| {
        format!(r#"{{"file":{},"line":{l},"col":1,"end_line":{el},"end_col":{ec},"text":{}}}"#, json_str(file), json_str(t))
    };
    if a.len() != b.len() {
        let last = a.len();
        return vec![edit(1, last, a.last().map_or(0, |x| x.len()) + 1, new)];
    }
    a.iter().zip(&b).enumerate().filter(|(_, (x, y))| x != y).map(|(i, (x, y))| edit(i + 1, i + 1, x.len() + 1, y)).collect()
}
