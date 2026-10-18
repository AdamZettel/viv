// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! SQLite inside Viv (crate::db): the `db_*`
//! builtins in the interpreter (SQLite linked into `viv`) and in C (the same
//! third_party/sqlite/sqlite3.c compiled beside the program), which must
//! print the same bytes and stop with the same words; the direct wasm
//! backend refuses them.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

fn viv_bin() -> &'static str {
    env!("CARGO_BIN_EXE_viv")
}

fn cc() -> String {
    std::env::var("CC").unwrap_or_else(|_| "cc".to_string())
}

/// sqlite3.c compiled once for the whole test run.
fn sqlite_obj() -> &'static Path {
    static OBJ: OnceLock<PathBuf> = OnceLock::new();
    OBJ.get_or_init(|| {
        let obj = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("viv_sqlite3_{}.o", std::process::id()));
        let src = viv::db::amalgamation_dir().join("sqlite3.c");
        let st = Command::new(cc()).args(["-c", "-O1", "-w"]).args(viv::db::SQLITE_FLAGS).arg(&src).arg("-o").arg(&obj).status().unwrap();
        assert!(st.success(), "compiling sqlite3.c");
        obj
    })
}

fn scratch(tag: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("viv_db_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[derive(Debug, PartialEq)]
struct Ran {
    code: Option<i32>,
    out: String,
    err: String,
}

fn ran(o: std::process::Output) -> Ran {
    Ran { code: o.status.code(), out: String::from_utf8_lossy(&o.stdout).into_owned(), err: String::from_utf8_lossy(&o.stderr).into_owned() }
}

/// Run each program in turn in the interpreter, in one directory, then each
/// compiled to C in a fresh one; the results of the last program, which
/// must agree.
fn engines(tag: &str, progs: &[&str]) -> Ran {
    let (di, dn) = (scratch(&format!("{tag}_i")), scratch(&format!("{tag}_n")));
    let mut last = None;
    for (k, src) in progs.iter().enumerate() {
        let name = format!("p{k}.viv");
        std::fs::write(di.join(&name), src).unwrap();
        std::fs::write(dn.join(&name), src).unwrap();
        let i = ran(Command::new(viv_bin()).current_dir(&di).args(["run", "-q"]).arg(&name).output().unwrap());
        let c = Command::new(viv_bin()).current_dir(&dn).arg(&name).args(["-o", "p.c"]).output().unwrap();
        assert!(c.status.success(), "{tag}: {}", String::from_utf8_lossy(&c.stderr));
        let c_text = std::fs::read_to_string(dn.join("p.c")).unwrap();
        assert!(viv::db::uses_sqlite(&c_text), "{tag}: the C does not mark its use of SQLite");
        let b = Command::new(cc()).current_dir(&dn).args(["-O2", "-ffp-contract=off", "p.c"]).arg(sqlite_obj()).args(["-lm", "-o", "p"]).output().unwrap();
        assert!(b.status.success(), "{tag}: cc: {}", String::from_utf8_lossy(&b.stderr));
        let n = ran(Command::new(dn.join("p")).current_dir(&dn).output().unwrap());
        assert_eq!(i, n, "{tag}: program {k}: native differs from the interpreter");
        last = Some(i);
    }
    last.unwrap()
}

fn prints(tag: &str, progs: &[&str], want: &str) {
    let r = engines(tag, progs);
    assert_eq!(r.code, Some(0), "{tag}: {r:?}");
    assert_eq!(r.out, want, "{tag}");
}

fn stops(tag: &str, progs: &[&str], msg: &str) {
    let r = engines(tag, progs);
    assert_eq!(r.code, Some(4), "{tag}: {r:?}");
    assert!(r.err.starts_with(&format!("error: {msg}")), "{tag}: {}", r.err);
}

const MAKE: &str = r#"
db = db_open("t.sqlite")
db_exec(db, "CREATE TABLE r (n INTEGER, e REAL, name TEXT UNIQUE)")
n: [Int] = [1, 2, 3]
db_stage(db, "n", n)
db_stage(db, "e", [0.5, 0.0 / 0.0, -2.25])
db_stage(db, "name", ["H2", "O'Brien'); DROP TABLE r; --", "Hé"])
print(db_insert(db, "r"))
"#;

#[test]
fn round_trip() {
    let read = r#"
db = db_open("t.sqlite")
print(db_col(db, "SELECT e FROM r ORDER BY n"))
print(db_col_int(db, "SELECT n FROM r ORDER BY n"))
print(db_col_str(db, "SELECT name FROM r ORDER BY n"))
print(db_col(db, "SELECT n FROM r ORDER BY n"))
print(db_col_str(db, "SELECT e FROM r WHERE e IS NOT NULL ORDER BY n"))
print(db_col(db, "SELECT json_extract('{\"a\": [1, 2.5]}', '$.a[1]')"))
print(db_col(db, "SELECT e FROM r WHERE 0"))
print(db_close(db))
"#;
    prints("round_trip", &[MAKE, read], "[0.5, nan, -2.25]\n[1, 2, 3]\n['H2', \"O'Brien'); DROP TABLE r; --\", 'Hé']\n[1, 2, 3]\n['0.5', '-2.25']\n[2.5]\n[]\n0\n");
}

#[test]
fn binding_is_not_pasting() {
    let read = r#"
db = db_open("t.sqlite")
evil = "x' OR '1'='1"
print(db_col_int(db, "SELECT n FROM r WHERE name = ?", [evil]))
print(db_col_int(db, "SELECT n FROM r WHERE name = ?", ["O'Brien'); DROP TABLE r; --"]))
print(db_col_int(db, "SELECT n FROM r WHERE e < ? AND n >= ?", [1.0, 2.0]))
print(db_col_int(db, "SELECT n FROM r WHERE n = ?", ["3"]))
print(db_exec(db, "UPDATE r SET e = ? WHERE n = ?", ["7.5", "2"]))
print(db_col(db, "SELECT e FROM r WHERE n = 2"))
print(db_exec(db, "UPDATE r SET e = ? WHERE n = ?", [0.0 / 0.0, 1.0]))
print(db_col(db, "SELECT count(*) FROM r WHERE e IS NULL"))
"#;
    prints("binding", &[MAKE, read], "[]\n[2]\n[3]\n[3]\n1\n[7.5]\n1\n[1]\n");
}

#[test]
fn null_rules() {
    stops("null_int", &[MAKE, "db = db_open(\"t.sqlite\")\nprint(db_col_int(db, \"SELECT CAST(e AS INTEGER) FROM r ORDER BY n\"))\n"], "db_col_int: row 1 holds NULL, not an Int");
    stops("null_str", &[MAKE, "db = db_open(\"t.sqlite\")\nprint(db_col_str(db, \"SELECT e FROM r ORDER BY n\"))\n"], "db_col_str: row 1 holds NULL, not text");
    stops("text_num", &[MAKE, "db = db_open(\"t.sqlite\")\nprint(db_col(db, \"SELECT name FROM r\"))\n"], "db_col: row 0 holds text, not a number");
    stops("frac_int", &[MAKE, "db = db_open(\"t.sqlite\")\nprint(db_col_int(db, \"SELECT e FROM r WHERE n = 1\"))\n"], "db_col_int: row 0 holds 0.5, not a whole number");
}

#[test]
fn misuse() {
    stops("count", &[MAKE, "db = db_open(\"t.sqlite\")\nprint(db_col(db, \"SELECT e FROM r WHERE n = ?\"))\n"], "db_col: the statement takes 1 parameters and was given 0");
    stops("cols", &[MAKE, "db = db_open(\"t.sqlite\")\nprint(db_col(db, \"SELECT n, e FROM r\"))\n"], "db_col: the query gives 2 columns; it reads one");
    stops("handle", &["print(db_col(3, \"SELECT 1\"))\n"], "db: 3 is not an open database");
    stops("sql", &["db = db_open(\"t.sqlite\")\nprint(db_exec(db, \"CREATE TABLEX\"))\n"], "db_exec: near \"TABLEX\": syntax error");
    stops("lengths", &["db = db_open(\"t.sqlite\")\ndb_exec(db, \"CREATE TABLE q (a, b)\")\ndb_stage(db, \"a\", [1.0, 2.0])\ndb_stage(db, \"b\", [1.0])\nprint(db_insert(db, \"q\"))\n"], "db_insert: column 'b' has 1 rows and column 'a' has 2");
}

#[test]
fn rollback_on_error() {
    // a script that fails half way leaves nothing behind
    let fail = "db = db_open(\"t.sqlite\")\nprint(db_exec(db, \"INSERT INTO r (n, name) VALUES (9, 'new'); INSERT INTO r (n, name) VALUES (10, 'H2')\"))\n";
    // an insert whose third row breaks UNIQUE writes no row
    let fail2 = "db = db_open(\"t.sqlite\")\nk: [Int] = [20, 21, 22]\ndb_stage(db, \"n\", k)\ndb_stage(db, \"name\", [\"a\", \"b\", \"H2\"])\nprint(db_insert(db, \"r\"))\n";
    let count = "db = db_open(\"t.sqlite\")\nprint(db_col_int(db, \"SELECT count(*) FROM r\"))\nprint(db_col_int(db, \"SELECT n FROM r ORDER BY n\"))\n";
    prints("rollback", &[MAKE, fail, fail2, count], "[3]\n[1, 2, 3]\n");
    stops("rollback_msg", &[MAKE, fail], "db_exec: UNIQUE constraint failed: r.name");
}

#[test]
fn wasm_refuses() {
    let dir = scratch("wasm");
    std::fs::write(dir.join("p.viv"), "db = db_open(\"t.sqlite\")\nprint(db)\n").unwrap();
    let r = ran(Command::new(viv_bin()).current_dir(&dir).args(["p.viv", "-o", "p.wasm"]).output().unwrap());
    assert_ne!(r.code, Some(0), "{r:?}");
    assert!(r.err.contains("not available in the browser yet"), "{r:?}");
}

#[test]
fn programs_without_db_have_no_sqlite() {
    let dir = scratch("plain");
    std::fs::write(dir.join("p.viv"), "s = \"a,b\"\nprint(split(s, \",\"))\nprint(read_text(\"p.viv\"))\n").unwrap();
    let c = Command::new(viv_bin()).current_dir(&dir).args(["p.viv", "-o", "p.c"]).output().unwrap();
    assert!(c.status.success());
    let text = std::fs::read_to_string(dir.join("p.c")).unwrap();
    assert!(!viv::db::uses_sqlite(&text) && !text.contains("sqlite3_"));
}
