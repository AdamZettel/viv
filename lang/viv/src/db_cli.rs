// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! `viv db`: the program database through the SQLite the
//! `viv` binary carries (crate::db).
//!
//! `query`, `queries` and `show --as graph|intent` run here, with the saved
//! queries of tools/vivdb/queries and the database at
//! `target/vivdb/viv.sqlite`, and print what `tools/vivdb/vivdb.py` prints,
//! byte for byte (tools/vivdb/test_vivdb.py compares them on the fixture).
//! `build` and `show --as code` (whose stored files are zlib-compressed)
//! still run the Python tool, which stays the reference.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use viv::db::{query_file, Value};

const USAGE: &str = "usage: viv db build [--db PATH] [--jobs N] [--paths P ...]   (runs tools/vivdb/vivdb.py)
       viv db query \"<SQL>\" | --saved NAME [--json] [--param k=v ...] [--db PATH]
       viv db queries
       viv db show <def> [--as code|graph|intent] [--all] [--db PATH]";

fn git(root: &Path, args: &[&str]) -> String {
    Command::new("git").arg("-C").arg(root).args(args).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default()
}

fn repo_root() -> PathBuf {
    PathBuf::from(git(Path::new("."), &["rev-parse", "--show-toplevel"]))
}

fn fail(m: impl std::fmt::Display) -> ExitCode {
    eprintln!("{m}");
    ExitCode::FAILURE
}

/// The saved queries, by name, sorted.
fn saved(root: &Path) -> Vec<(String, PathBuf)> {
    let dir = root.join("tools/vivdb/queries");
    let mut v: Vec<(String, PathBuf)> = std::fs::read_dir(&dir)
        .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "sql")).map(|p| (p.file_stem().unwrap().to_string_lossy().into_owned(), p)).collect())
        .unwrap_or_default();
    v.sort();
    v
}

/// Python's `repr` of a float.
fn py_float(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    let e = format!("{x:e}");
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let sign = if neg { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let n = digits.len() as i32;
        let s = if exp >= n - 1 {
            format!("{digits}{}.0", "0".repeat((exp - n + 1) as usize))
        } else if exp >= 0 {
            format!("{}.{}", &digits[..(exp + 1) as usize], &digits[(exp + 1) as usize..])
        } else {
            format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
        };
        format!("{sign}{s}")
    } else {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits };
        format!("{sign}{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

/// Python's `str` of a value, as `print_table` writes a cell.
fn cell(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::Int(i) => i.to_string(),
        Value::Real(x) => py_float(*x),
        Value::Text(s) => s.clone(),
        Value::Blob(b) => format!("b{:?}", String::from_utf8_lossy(b)),
    }
}

/// `json.dumps` of a string (ensure_ascii).
fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            '\u{8}' => o.push_str("\\b"),
            '\u{c}' => o.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    o.push_str(&format!("\\u{:04x}", u));
                }
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn json_val(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Int(i) => i.to_string(),
        Value::Real(x) if x.is_nan() => "NaN".into(),
        Value::Real(x) if x.is_infinite() => if *x > 0.0 { "Infinity" } else { "-Infinity" }.into(),
        Value::Real(x) => py_float(*x),
        Value::Text(s) => json_str(s),
        Value::Blob(b) => json_str(&String::from_utf8_lossy(b)),
    }
}

/// `print_table` of vivdb.py.
fn print_table(cols: &[String], rows: &[Vec<Value>], as_json: bool) {
    if as_json {
        if rows.is_empty() {
            println!("[]");
            return;
        }
        let mut out = String::from("[\n");
        for (k, r) in rows.iter().enumerate() {
            // dict(zip(cols, r)): a repeated name keeps its first place and its last value
            let mut kv: Vec<(&str, &Value)> = Vec::new();
            for (c, v) in cols.iter().zip(r) {
                match kv.iter_mut().find(|(n, _)| *n == c.as_str()) {
                    Some(slot) => slot.1 = v,
                    None => kv.push((c, v)),
                }
            }
            if kv.is_empty() {
                out.push_str(" {}");
            } else {
                out.push_str(" {\n");
                let body: Vec<String> = kv.iter().map(|(c, v)| format!("  {}: {}", json_str(c), json_val(v))).collect();
                out.push_str(&body.join(",\n"));
                out.push_str("\n }");
            }
            out.push_str(if k + 1 < rows.len() { ",\n" } else { "\n" });
        }
        out.push(']');
        println!("{out}");
        return;
    }
    if cols.is_empty() {
        return;
    }
    let cells: Vec<Vec<String>> = rows.iter().map(|r| r.iter().map(cell).collect()).collect();
    let n = |s: &str| s.chars().count();
    let width: Vec<usize> = cols.iter().enumerate().map(|(i, c)| cells.iter().map(|r| n(&r[i])).chain([n(c)]).max().unwrap().min(60)).collect();
    let line = |vals: &[String]| -> String {
        let parts: Vec<String> = vals
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let t: String = v.chars().take(60).collect();
                let pad = width[i].saturating_sub(n(&t));
                format!("{t}{}", " ".repeat(pad))
            })
            .collect();
        parts.join("  ").trim_end().to_string()
    };
    println!("{}", line(cols));
    println!("{}", line(&width.iter().map(|w| "-".repeat(*w)).collect::<Vec<_>>()));
    for r in &cells {
        println!("{}", line(r));
    }
    println!("({} row{})", rows.len(), if rows.len() != 1 { "s" } else { "" });
}

/// The `:name` parameters a query names (vivdb.py's `(?<!:):([A-Za-z_]\w*)`).
fn param_names(sql: &str) -> Vec<String> {
    let b = sql.as_bytes();
    let mut out: Vec<String> = Vec::new();
    for i in 0..b.len() {
        if b[i] == b':' && (i == 0 || b[i - 1] != b':') && i + 1 < b.len() && (b[i + 1].is_ascii_alphabetic() || b[i + 1] == b'_') {
            let mut j = i + 1;
            while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
                j += 1;
            }
            let name = sql[i + 1..j].to_string();
            if !out.contains(&name) {
                out.push(name);
            }
        }
    }
    out
}

struct Opts {
    db: Option<String>,
    json: bool,
    all: bool,
    saved: Option<String>,
    as_: String,
    params: Vec<(String, String)>,
    pos: Vec<String>,
}

fn parse(args: &[String]) -> Result<Opts, String> {
    let mut o = Opts { db: None, json: false, all: false, saved: None, as_: "code".into(), params: Vec::new(), pos: Vec::new() };
    let mut i = 0;
    while i < args.len() {
        let next = |i: usize| args.get(i + 1).cloned().ok_or_else(|| USAGE.to_string());
        match args[i].as_str() {
            "--db" => { o.db = Some(next(i)?); i += 1; }
            "--saved" => { o.saved = Some(next(i)?); i += 1; }
            "--as" => { o.as_ = next(i)?; i += 1; }
            "--param" => {
                let p = next(i)?;
                let (k, v) = p.split_once('=').ok_or_else(|| format!("vivdb: --param takes k=v, not {p}"))?;
                o.params.retain(|(n, _)| n != k);
                o.params.push((k.to_string(), v.to_string()));
                i += 1;
            }
            "--json" => o.json = true,
            "--all" => o.all = true,
            a if a.starts_with("--") => return Err(USAGE.to_string()),
            a => o.pos.push(a.to_string()),
        }
        i += 1;
    }
    Ok(o)
}

fn open_path(root: &Path, o: &Opts) -> Result<String, String> {
    let db = o.db.clone().unwrap_or_else(|| root.join("target/vivdb/viv.sqlite").display().to_string());
    if !Path::new(&db).exists() {
        return Err(format!("vivdb: no database at {db}; run `vivdb.py build` first"));
    }
    Ok(db)
}

fn stale_note(root: &Path, db: &str) {
    if let Ok((_, rows)) = query_file(db, "SELECT value FROM meta WHERE key = 'commit'", &[]) {
        let built = match rows.first().and_then(|r| r.first()) {
            Some(Value::Text(s)) => s.clone(),
            _ => "?".into(),
        };
        let head = git(root, &["rev-parse", "HEAD"]);
        if built != head {
            eprintln!("note: the database was built at {}, HEAD is {}; rebuild for current answers", &built[..built.len().min(8)], &head[..head.len().min(8)]);
        }
    }
}

fn query(root: &Path, o: &Opts) -> Result<(), String> {
    let db = open_path(root, o)?;
    let sql = match (&o.saved, o.pos.first()) {
        (Some(name), _) => {
            let qs = saved(root);
            let Some((_, p)) = qs.iter().find(|(n, _)| n == name) else {
                return Err(format!("vivdb: no saved query `{name}`; saved: {}", qs.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", ")));
            };
            std::fs::read_to_string(p).map_err(|e| e.to_string())?
        }
        (None, Some(sql)) => sql.clone(),
        _ => return Err("vivdb: query takes SQL or --saved NAME".into()),
    };
    let names = param_names(&sql);
    let mut missing: Vec<String> = names.iter().filter(|n| !o.params.iter().any(|(k, _)| k == *n)).map(|n| format!("{n}=...")).collect();
    if !missing.is_empty() {
        missing.sort();
        return Err(format!("vivdb: the query needs --param {}", missing.join(" ")));
    }
    let bound: Vec<(String, String)> = o.params.iter().filter(|(k, _)| names.contains(k)).cloned().collect();
    let (cols, rows) = query_file(&db, &sql, &bound)?;
    stale_note(root, &db);
    print_table(&cols, &rows, o.json);
    Ok(())
}

fn text(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        other => cell(other),
    }
}

fn show(root: &Path, o: &Opts, args: &[String]) -> Result<(), String> {
    if o.as_ == "code" {
        // the stored files are zlib-compressed: the Python tool reads them
        return python(root, args);
    }
    if o.as_ != "graph" && o.as_ != "intent" {
        return Err(USAGE.into());
    }
    let Some(r) = o.pos.first() else { return Err(USAGE.into()) };
    let db = open_path(root, o)?;
    let q = |sql: &str, ps: &[(&str, &str)]| -> Result<Vec<Vec<Value>>, String> {
        let named: Vec<(String, String)> = ps.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        query_file(&db, sql, &named).map(|x| x.1)
    };
    let mut sites = Vec::new();
    for sep in [':', '.'] {
        if let Some((m, n)) = r.rsplit_once(sep) {
            sites = q("SELECT module, name FROM def_sites WHERE module = :m AND name = :n", &[("m", m), ("n", n)])?;
            if !sites.is_empty() {
                break;
            }
        }
    }
    if sites.is_empty() {
        sites = q("SELECT module, name FROM def_sites WHERE name = :n ORDER BY module", &[("n", r)])?;
    }
    if sites.is_empty() {
        return Err(format!("vivdb: no def `{r}`"));
    }
    if sites.len() > 1 && !o.all {
        eprintln!("`{r}` is defined in {} modules; showing the first (--all for every one): {}", sites.len(), sites.iter().map(|s| text(&s[0])).collect::<Vec<_>>().join(", "));
        sites.truncate(1);
    }
    for s in &sites {
        let (module, name) = (text(&s[0]), text(&s[1]));
        let mn: &[(&str, &str)] = &[("m", &module), ("n", &name)];
        if o.as_ == "graph" {
            println!("{module}.{name}");
            for c in q("SELECT DISTINCT caller_module, caller, caller_kind FROM calls WHERE callee_module = :m AND callee = :n ORDER BY 1, 2", mn)? {
                let (cm, cn, k) = (text(&c[0]), text(&c[1]), text(&c[2]));
                println!("  <- {}{}", if k == "test" { cn } else { format!("{cm}.{cn}") }, if k == "def" { String::new() } else { format!("  ({k})") });
            }
            for c in q("SELECT callee_name, target, MIN(line), value FROM calls WHERE caller_module = :m AND caller = :n GROUP BY callee_name, target ORDER BY 3", mn)? {
                let value = !matches!(c[3], Value::Null | Value::Int(0));
                println!("  -> {}  [{}{}]  line {}", text(&c[0]), text(&c[1]), if value { ", as a value" } else { "" }, text(&c[2]));
            }
            for c in q("SELECT step, heading, depth, line_start, line_end FROM steps WHERE module = :m AND def_name = :n ORDER BY ord", mn)? {
                let depth = if let Value::Int(d) = c[2] { d.max(0) as usize } else { 0 };
                let step = if matches!(&c[0], Value::Null) || text(&c[0]).is_empty() { "-".to_string() } else { text(&c[0]) };
                println!("  {}step {step}: {}  (lines {}-{})", "  ".repeat(depth), text(&c[1]), text(&c[3]), text(&c[4]));
            }
        } else {
            let st = q("SELECT state, stamp, source FROM intent WHERE module = :m AND def_name = :n AND step_name = ''", mn)?;
            let ir = q("SELECT ir_hash FROM def_sites WHERE module = :m AND name = :n", mn)?;
            let mut line = format!("{module}.{name}: intent {}", st.first().map_or("missing".to_string(), |r| text(&r[0])));
            if let Some(r) = st.first() {
                if !matches!(&r[1], Value::Null) && !text(&r[1]).is_empty() {
                    line.push_str(&format!(", stamped @{}", text(&r[1])));
                }
                if text(&r[2]) != "none" {
                    line.push_str(&format!(" ({})", text(&r[2])));
                }
            }
            if let Some(r) = ir.first() {
                if !matches!(&r[0], Value::Null) && !text(&r[0]).is_empty() {
                    line.push_str(&format!(", body {}", text(&r[0])));
                }
            }
            println!("{line}");
            for c in q("SELECT label, text, line FROM docs WHERE module = :m AND def_name = :n ORDER BY line", mn)? {
                println!("  {}: {}", text(&c[0]), text(&c[1]));
            }
            for c in q("SELECT step, heading, depth FROM steps WHERE module = :m AND def_name = :n ORDER BY ord", mn)? {
                let depth = if let Value::Int(d) = c[2] { d.max(0) as usize } else { 0 };
                let step = if matches!(&c[0], Value::Null) || text(&c[0]).is_empty() { "-".to_string() } else { text(&c[0]) };
                println!("  {}{step}: {}", "  ".repeat(depth), text(&c[1]));
            }
        }
    }
    Ok(())
}

/// Run the Python tool with these arguments (`build`, `show --as code`).
fn python(root: &Path, args: &[String]) -> Result<(), String> {
    let py = std::env::var("PYTHON").unwrap_or_else(|_| "python3".into());
    let st = Command::new(&py).arg(root.join("tools/vivdb/vivdb.py")).args(args).status().map_err(|e| format!("error: cannot run {py}: {e}"))?;
    if st.success() { Ok(()) } else { Err(String::new()) }
}

pub fn main(args: &[String]) -> ExitCode {
    let root = repo_root();
    let Some(verb) = args.first() else { return fail(USAGE) };
    let done = match verb.as_str() {
        "build" => python(&root, args),
        "queries" => {
            for (name, p) in saved(&root) {
                let text = std::fs::read_to_string(&p).unwrap_or_default();
                let first = text.lines().find(|l| l.starts_with("-- ")).map_or("", |l| l[3..].trim());
                println!("{:28} {first}", name);
            }
            Ok(())
        }
        "query" | "show" => match parse(&args[1..]) {
            Ok(o) if verb == "query" => query(&root, &o),
            Ok(o) => show(&root, &o, args),
            Err(e) => Err(e),
        },
        "-h" | "--help" => {
            println!("{USAGE}");
            Ok(())
        }
        _ => Err(USAGE.into()),
    };
    match done {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) if e.is_empty() => ExitCode::FAILURE,
        Err(e) => fail(e),
    }
}
