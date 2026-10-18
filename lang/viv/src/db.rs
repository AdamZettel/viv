// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! SQLite inside Viv: the `db_*` builtins of
//! the interpreter and the database under `viv db`.
//!
//! The library is `third_party/sqlite/sqlite3.c`, compiled by build.rs and
//! linked into the `viv` binary; this file is its small FFI and the rules of
//! the builtins on plain Rust values. `interp/db.c` is the C engine's twin,
//! rule for rule and message for message; `tests/db.rs` holds the two to the
//! same outputs.
//!
//! The rules:
//! - a database is a number, the handle `db_open` gives (1, 2, ...);
//! - parameters are bound (`?` in the SQL), a list of strings as TEXT or an
//!   array of numbers as REAL (NaN as NULL); a statement with parameters is
//!   one statement, and the count of `?` must match;
//! - `db_exec` runs its SQL inside a savepoint: a statement that fails
//!   leaves the database as it was before the call (so the SQL itself may
//!   not hold BEGIN or COMMIT);
//! - a query (`db_col`, `db_col_int`, `db_col_str`) is one statement giving
//!   one column. NULL reads as NaN in `db_col`, and is an error in
//!   `db_col_int` and `db_col_str` (an Int and a string have no NULL: select
//!   `COALESCE(x, k)`). Text where a number goes is an error;
//! - `db_stage(db, name, column)` sets a column aside and `db_insert(db,
//!   table)` writes every staged column as rows in one transaction (NaN as
//!   NULL), then clears them. Names are quoted as identifiers.
//!
//! Errors from SQLite are VivError (`ErrKind::Error`) with SQLite's message;
//! a misuse (a bad handle, a wrong count, a NULL in an Int column) is
//! ValueError.

use crate::errkind::{kinded, ErrKind};

/// The compiler flags SQLite is built with, everywhere (third_party/sqlite/README.md).
pub const SQLITE_FLAGS: &[&str] = &["-DSQLITE_OMIT_LOAD_EXTENSION", "-DSQLITE_DEFAULT_FOREIGN_KEYS=1"];

/// Where `sqlite3.c` is: `$VIV_SQLITE_DIR`, or the repository's copy.
pub fn amalgamation_dir() -> std::path::PathBuf {
    match std::env::var_os("VIV_SQLITE_DIR") {
        Some(d) => d.into(),
        None => std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../third_party/sqlite"),
    }
}

/// Whether a program's C uses the `db_*` builtins (the `str_db` runtime
/// fragment is in it).
pub fn uses_sqlite(c: &str) -> bool {
    c.contains("VIV_USES_SQLITE")
}

/// An exported program's Python wrapper, building `sqlite3.c` (copied
/// beside it) into the library with the program.
pub fn py_with_sqlite(py: &str) -> String {
    let flags: Vec<String> = SQLITE_FLAGS.iter().map(|f| format!("{f:?}")).collect();
    py.replace(
        "[_src, \"-o\", path, \"-lm\"]",
        &format!("[_src, _os.path.join(_here, \"sqlite3.c\"), {}, \"-o\", path, \"-lm\"]", flags.join(", ")),
    )
}

/// The parameters of a statement.
pub enum Params {
    None,
    Text(Vec<String>),
    Real(Vec<f64>),
}

/// A staged column.
pub enum Col {
    Real(Vec<f64>),
    Int(Vec<i64>),
    Text(Vec<String>),
}

impl Col {
    fn len(&self) -> usize {
        match self {
            Col::Real(v) => v.len(),
            Col::Int(v) => v.len(),
            Col::Text(v) => v.len(),
        }
    }
}

/// One value read from a row, for `viv db query`.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Int(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

/// A number in a message: a whole number without a fraction.
fn num_text(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 { format!("{}", x as i64) } else { format!("{x}") }
}

fn err(m: String) -> String {
    kinded(ErrKind::Error, m)
}
fn verr(m: String) -> String {
    kinded(ErrKind::Value, m)
}

#[cfg(not(target_arch = "wasm32"))]
mod ffi {
    use std::os::raw::{c_char, c_double, c_int, c_void};
    #[repr(C)]
    pub struct Sqlite3 {
        _p: [u8; 0],
    }
    #[repr(C)]
    pub struct Stmt {
        _p: [u8; 0],
    }
    pub const OK: c_int = 0;
    pub const ROW: c_int = 100;
    pub const DONE: c_int = 101;
    pub const INTEGER: c_int = 1;
    pub const FLOAT: c_int = 2;
    pub const TEXT: c_int = 3;
    pub const BLOB: c_int = 4;
    pub const NULL: c_int = 5;
    /// SQLITE_TRANSIENT: SQLite copies the bytes
    pub fn transient() -> Option<unsafe extern "C" fn(*mut c_void)> {
        unsafe { std::mem::transmute::<isize, Option<unsafe extern "C" fn(*mut c_void)>>(-1) }
    }
    extern "C" {
        pub fn sqlite3_open_v2(f: *const c_char, db: *mut *mut Sqlite3, flags: c_int, vfs: *const c_char) -> c_int;
        pub fn sqlite3_close_v2(db: *mut Sqlite3) -> c_int;
        pub fn sqlite3_errmsg(db: *mut Sqlite3) -> *const c_char;
        pub fn sqlite3_exec(db: *mut Sqlite3, sql: *const c_char, cb: *const c_void, arg: *mut c_void, err: *mut *mut c_char) -> c_int;
        pub fn sqlite3_prepare_v2(db: *mut Sqlite3, sql: *const c_char, n: c_int, st: *mut *mut Stmt, tail: *mut *const c_char) -> c_int;
        pub fn sqlite3_finalize(st: *mut Stmt) -> c_int;
        pub fn sqlite3_reset(st: *mut Stmt) -> c_int;
        pub fn sqlite3_step(st: *mut Stmt) -> c_int;
        pub fn sqlite3_bind_parameter_count(st: *mut Stmt) -> c_int;
        pub fn sqlite3_bind_parameter_name(st: *mut Stmt, i: c_int) -> *const c_char;
        pub fn sqlite3_bind_text(st: *mut Stmt, i: c_int, s: *const c_char, n: c_int, d: Option<unsafe extern "C" fn(*mut c_void)>) -> c_int;
        pub fn sqlite3_bind_double(st: *mut Stmt, i: c_int, x: c_double) -> c_int;
        pub fn sqlite3_bind_int64(st: *mut Stmt, i: c_int, x: i64) -> c_int;
        pub fn sqlite3_bind_null(st: *mut Stmt, i: c_int) -> c_int;
        pub fn sqlite3_column_count(st: *mut Stmt) -> c_int;
        pub fn sqlite3_column_name(st: *mut Stmt, i: c_int) -> *const c_char;
        pub fn sqlite3_column_type(st: *mut Stmt, i: c_int) -> c_int;
        pub fn sqlite3_column_double(st: *mut Stmt, i: c_int) -> c_double;
        pub fn sqlite3_column_int64(st: *mut Stmt, i: c_int) -> i64;
        pub fn sqlite3_column_text(st: *mut Stmt, i: c_int) -> *const u8;
        pub fn sqlite3_column_blob(st: *mut Stmt, i: c_int) -> *const c_void;
        pub fn sqlite3_column_bytes(st: *mut Stmt, i: c_int) -> c_int;
        pub fn sqlite3_total_changes(db: *mut Sqlite3) -> c_int;
        pub fn sqlite3_libversion() -> *const c_char;
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::*;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::ffi::*;
    use super::*;
    use std::cell::RefCell;
    use std::ffi::{CStr, CString};
    use std::ptr;

    /// An open database: its connection and its staged columns.
    struct Conn {
        db: *mut Sqlite3,
        staged: Vec<(String, Col)>,
    }

    thread_local! {
        static CONNS: RefCell<Vec<Option<Conn>>> = const { RefCell::new(Vec::new()) };
    }

    /// The SQLite library's version, as `sqlite3_libversion` gives it.
    pub fn version() -> String {
        unsafe { CStr::from_ptr(sqlite3_libversion()).to_string_lossy().into_owned() }
    }

    fn cstr(s: &str, what: &str) -> Result<CString, String> {
        CString::new(s).map_err(|_| verr(format!("{what}: a NUL character in the text")))
    }

    fn msg(db: *mut Sqlite3) -> String {
        unsafe { CStr::from_ptr(sqlite3_errmsg(db)).to_string_lossy().into_owned() }
    }

    fn with<T>(h: f64, f: impl FnOnce(&mut Conn) -> Result<T, String>) -> Result<T, String> {
        CONNS.with(|c| {
            let mut c = c.borrow_mut();
            let k = h as usize;
            if !(h >= 1.0 && h.fract() == 0.0 && k <= c.len()) || c[k - 1].is_none() {
                return Err(verr(format!("db: {} is not an open database", num_text(h))));
            }
            f(c[k - 1].as_mut().unwrap())
        })
    }

    /// Open (or create) the database file `path`; its handle.
    pub fn open(path: &str) -> Result<f64, String> {
        let p = cstr(path, "db_open")?;
        let mut db = ptr::null_mut();
        // READWRITE | CREATE
        let rc = unsafe { sqlite3_open_v2(p.as_ptr(), &mut db, 0x2 | 0x4, ptr::null()) };
        if rc != OK {
            let m = if db.is_null() { "out of memory".to_string() } else { msg(db) };
            unsafe { sqlite3_close_v2(db) };
            return Err(err(format!("db_open: {}: {m}", crate::interp::strings::repr(path))));
        }
        CONNS.with(|c| {
            let mut c = c.borrow_mut();
            c.push(Some(Conn { db, staged: Vec::new() }));
            Ok(c.len() as f64)
        })
    }

    /// Close the database `h`; 0.
    pub fn close(h: f64) -> Result<f64, String> {
        with(h, |c| {
            unsafe { sqlite3_close_v2(c.db) };
            Ok(())
        })?;
        CONNS.with(|c| c.borrow_mut()[h as usize - 1] = None);
        Ok(0.0)
    }

    fn plain(db: *mut Sqlite3, sql: &str) {
        let s = CString::new(sql).unwrap();
        unsafe { sqlite3_exec(db, s.as_ptr(), ptr::null(), ptr::null_mut(), ptr::null_mut()) };
    }

    /// Run `f` inside a savepoint: released when it succeeds, rolled back
    /// when it fails.
    fn atomic<T>(db: *mut Sqlite3, f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
        plain(db, "SAVEPOINT viv_db");
        let r = f();
        if r.is_err() {
            plain(db, "ROLLBACK TO viv_db");
        }
        plain(db, "RELEASE viv_db");
        r
    }

    /// A prepared statement, finalized when dropped.
    struct St(*mut Stmt);
    impl Drop for St {
        fn drop(&mut self) {
            unsafe { sqlite3_finalize(self.0) };
        }
    }

    /// Prepare the statement at the start of `sql`; the statement (None for
    /// only whitespace or a comment) and the rest of the text.
    fn prepare<'a>(db: *mut Sqlite3, sql: &'a CStr, who: &str) -> Result<(Option<St>, &'a CStr), String> {
        let mut st = ptr::null_mut();
        let mut tail: *const std::os::raw::c_char = ptr::null();
        let rc = unsafe { sqlite3_prepare_v2(db, sql.as_ptr(), -1, &mut st, &mut tail) };
        if rc != OK {
            return Err(err(format!("{who}: {}", msg(db))));
        }
        let rest = unsafe { CStr::from_ptr(tail) };
        Ok(((!st.is_null()).then_some(St(st)), rest))
    }

    fn blank(s: &CStr) -> bool {
        s.to_bytes().iter().all(|b| b.is_ascii_whitespace() || *b == b';')
    }

    fn bind(st: &St, p: &Params, who: &str) -> Result<(), String> {
        let want = unsafe { sqlite3_bind_parameter_count(st.0) } as usize;
        let n = match p {
            Params::None => 0,
            Params::Text(v) => v.len(),
            Params::Real(v) => v.len(),
        };
        if n != want {
            return Err(verr(format!("{who}: the statement takes {want} parameters and was given {n}")));
        }
        for i in 0..n {
            let k = (i + 1) as i32;
            unsafe {
                match p {
                    Params::Text(v) => {
                        sqlite3_bind_text(st.0, k, v[i].as_ptr() as *const _, v[i].len() as i32, transient());
                    }
                    Params::Real(v) if v[i].is_nan() => {
                        sqlite3_bind_null(st.0, k);
                    }
                    Params::Real(v) => {
                        sqlite3_bind_double(st.0, k, v[i]);
                    }
                    Params::None => {}
                }
            }
        }
        Ok(())
    }

    /// Step a statement to its end, calling `row` on each row.
    fn run_rows(db: *mut Sqlite3, st: &St, who: &str, mut row: impl FnMut(*mut Stmt) -> Result<(), String>) -> Result<(), String> {
        loop {
            match unsafe { sqlite3_step(st.0) } {
                ROW => row(st.0)?,
                DONE => return Ok(()),
                _ => return Err(err(format!("{who}: {}", msg(db)))),
            }
        }
    }

    /// Run `sql` (several statements without parameters, or one with them);
    /// the rows it changed.
    pub fn exec(h: f64, sql: &str, p: &Params) -> Result<f64, String> {
        let s = cstr(sql, "db_exec")?;
        with(h, |c| {
            let db = c.db;
            let before = unsafe { sqlite3_total_changes(db) };
            atomic(db, || {
                let mut rest: &CStr = &s;
                loop {
                    if blank(rest) {
                        break;
                    }
                    let (st, tail) = prepare(db, rest, "db_exec")?;
                    if let Some(st) = st {
                        bind(&st, p, "db_exec")?;
                        if !matches!(p, Params::None) && !blank(tail) {
                            return Err(verr("db_exec: SQL with parameters is one statement".to_string()));
                        }
                        run_rows(db, &st, "db_exec", |_| Ok(()))?;
                    }
                    rest = tail;
                }
                Ok(())
            })?;
            Ok((unsafe { sqlite3_total_changes(db) } - before) as f64)
        })
    }

    /// The one statement of a query, prepared and bound.
    fn query(db: *mut Sqlite3, s: &CStr, p: &Params, who: &str) -> Result<St, String> {
        let (st, tail) = prepare(db, s, who)?;
        let Some(st) = st else { return Err(verr(format!("{who}: no statement"))) };
        if !blank(tail) {
            return Err(verr(format!("{who}: a query is one statement")));
        }
        bind(&st, p, who)?;
        Ok(st)
    }

    fn one_col(st: &St, who: &str) -> Result<(), String> {
        let n = unsafe { sqlite3_column_count(st.0) };
        if n != 1 {
            return Err(verr(format!("{who}: the query gives {n} columns; it reads one")));
        }
        Ok(())
    }

    fn type_name(t: i32) -> &'static str {
        match t {
            TEXT => "text",
            BLOB => "a blob",
            NULL => "NULL",
            _ => "a number",
        }
    }

    /// The query's one column as numbers: NULL is NaN.
    pub fn col_real(h: f64, sql: &str, p: &Params) -> Result<Vec<f64>, String> {
        let s = cstr(sql, "db_col")?;
        with(h, |c| {
            let st = query(c.db, &s, p, "db_col")?;
            one_col(&st, "db_col")?;
            let mut out = Vec::new();
            run_rows(c.db, &st, "db_col", |st| {
                let t = unsafe { sqlite3_column_type(st, 0) };
                out.push(match t {
                    NULL => f64::NAN,
                    INTEGER | FLOAT => unsafe { sqlite3_column_double(st, 0) },
                    _ => return Err(verr(format!("db_col: row {} holds {}, not a number", out.len(), type_name(t)))),
                });
                Ok(())
            })?;
            Ok(out)
        })
    }

    /// The query's one column as Ints: a REAL must be a whole number, and a
    /// NULL is an error.
    pub fn col_int(h: f64, sql: &str, p: &Params) -> Result<Vec<i64>, String> {
        let s = cstr(sql, "db_col_int")?;
        with(h, |c| {
            let st = query(c.db, &s, p, "db_col_int")?;
            one_col(&st, "db_col_int")?;
            let mut out = Vec::new();
            run_rows(c.db, &st, "db_col_int", |st| {
                let t = unsafe { sqlite3_column_type(st, 0) };
                let x = unsafe { sqlite3_column_double(st, 0) };
                out.push(match t {
                    INTEGER => unsafe { sqlite3_column_int64(st, 0) },
                    FLOAT if x.fract() == 0.0 && x.abs() < 9.2e18 => x as i64,
                    FLOAT => return Err(verr(format!("db_col_int: row {} holds {}, not a whole number", out.len(), num_text(x)))),
                    _ => return Err(verr(format!("db_col_int: row {} holds {}, not an Int", out.len(), type_name(t)))),
                });
                Ok(())
            })?;
            Ok(out)
        })
    }

    /// The query's one column as text; a NULL is an error.
    pub fn col_text(h: f64, sql: &str, p: &Params) -> Result<Vec<String>, String> {
        let s = cstr(sql, "db_col_str")?;
        with(h, |c| {
            let st = query(c.db, &s, p, "db_col_str")?;
            one_col(&st, "db_col_str")?;
            let mut out = Vec::new();
            run_rows(c.db, &st, "db_col_str", |st| {
                let t = unsafe { sqlite3_column_type(st, 0) };
                if t == NULL || t == BLOB {
                    return Err(verr(format!("db_col_str: row {} holds {}, not text", out.len(), type_name(t))));
                }
                out.push(text_at(st, 0));
                Ok(())
            })?;
            Ok(out)
        })
    }

    fn text_at(st: *mut Stmt, i: i32) -> String {
        unsafe {
            let p = sqlite3_column_text(st, i);
            let n = sqlite3_column_bytes(st, i) as usize;
            if p.is_null() {
                return String::new();
            }
            String::from_utf8_lossy(std::slice::from_raw_parts(p, n)).into_owned()
        }
    }

    /// Set a column aside for the next `insert`; the rows staged.
    pub fn stage(h: f64, name: &str, col: Col) -> Result<f64, String> {
        with(h, |c| {
            if c.staged.iter().any(|(n, _)| n == name) {
                return Err(verr(format!("db_stage: column {} is staged already", crate::interp::strings::repr(name))));
            }
            let n = col.len();
            c.staged.push((name.to_string(), col));
            Ok(n as f64)
        })
    }

    fn quote_ident(s: &str) -> String {
        format!("\"{}\"", s.replace('"', "\"\""))
    }

    /// Write the staged columns into `table` as rows, in one transaction;
    /// the rows written. The stage is cleared either way.
    pub fn insert(h: f64, table: &str) -> Result<f64, String> {
        with(h, |c| {
            let staged = std::mem::take(&mut c.staged);
            if staged.is_empty() {
                return Err(verr("db_insert: no columns staged".to_string()));
            }
            let rows = staged[0].1.len();
            if let Some((n, col)) = staged.iter().find(|(_, col)| col.len() != rows) {
                return Err(verr(format!("db_insert: column {} has {} rows and column {} has {rows}", crate::interp::strings::repr(n), col.len(), crate::interp::strings::repr(&staged[0].0))));
            }
            let names: Vec<String> = staged.iter().map(|(n, _)| quote_ident(n)).collect();
            let marks = vec!["?"; staged.len()].join(", ");
            let sql = format!("INSERT INTO {} ({}) VALUES ({marks})", quote_ident(table), names.join(", "));
            let s = cstr(&sql, "db_insert")?;
            let db = c.db;
            atomic(db, || {
                let (st, _) = prepare(db, &s, "db_insert")?;
                let st = st.unwrap();
                for r in 0..rows {
                    unsafe {
                        sqlite3_reset(st.0);
                        for (j, (_, col)) in staged.iter().enumerate() {
                            let k = (j + 1) as i32;
                            match col {
                                Col::Real(v) if v[r].is_nan() => sqlite3_bind_null(st.0, k),
                                Col::Real(v) => sqlite3_bind_double(st.0, k, v[r]),
                                Col::Int(v) => sqlite3_bind_int64(st.0, k, v[r]),
                                Col::Text(v) => sqlite3_bind_text(st.0, k, v[r].as_ptr() as *const _, v[r].len() as i32, transient()),
                            };
                        }
                    }
                    run_rows(db, &st, "db_insert", |_| Ok(()))?;
                }
                Ok(rows as f64)
            })
        })
    }

    /// Every row of `sql` on the database file `path` opened read-only, its
    /// parameters bound by name as text (`:k` from `("k", v)`), for `viv db`:
    /// the column names and the rows. A name with no value is bound NULL.
    pub fn query_file(path: &str, sql: &str, named: &[(String, String)]) -> Result<(Vec<String>, Vec<Vec<Value>>), String> {
        let p = cstr(path, "db")?;
        let mut db = ptr::null_mut();
        let rc = unsafe { sqlite3_open_v2(p.as_ptr(), &mut db, 0x1, ptr::null()) };
        if rc != OK {
            let m = if db.is_null() { "out of memory".to_string() } else { msg(db) };
            unsafe { sqlite3_close_v2(db) };
            return Err(format!("{path}: {m}"));
        }
        let r = (|| {
            let s = cstr(sql, "query")?;
            let (st, _) = prepare(db, &s, "query")?;
            let Some(st) = st else { return Ok((Vec::new(), Vec::new())) };
            let np = unsafe { sqlite3_bind_parameter_count(st.0) };
            for i in 1..=np {
                let name = unsafe { sqlite3_bind_parameter_name(st.0, i) };
                if name.is_null() {
                    continue;
                }
                let name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
                if let Some((_, v)) = named.iter().find(|(k, _)| *k == name[1..]) {
                    unsafe { sqlite3_bind_text(st.0, i, v.as_ptr() as *const _, v.len() as i32, transient()) };
                }
            }
            collect(db, &st)
        })();
        unsafe { sqlite3_close_v2(db) };
        r
    }

    fn collect(db: *mut Sqlite3, st: &St) -> Result<(Vec<String>, Vec<Vec<Value>>), String> {
        let n = unsafe { sqlite3_column_count(st.0) };
        let names = (0..n).map(|i| unsafe { CStr::from_ptr(sqlite3_column_name(st.0, i)).to_string_lossy().into_owned() }).collect();
        let mut out = Vec::new();
        run_rows(db, st, "query", |st| {
            out.push(
                (0..n)
                    .map(|i| unsafe {
                        match sqlite3_column_type(st, i) {
                            NULL => Value::Null,
                            INTEGER => Value::Int(sqlite3_column_int64(st, i)),
                            FLOAT => Value::Real(sqlite3_column_double(st, i)),
                            TEXT => Value::Text(text_at(st, i)),
                            _ => {
                                let p = sqlite3_column_blob(st, i) as *const u8;
                                let k = sqlite3_column_bytes(st, i) as usize;
                                Value::Blob(if p.is_null() { Vec::new() } else { std::slice::from_raw_parts(p, k).to_vec() })
                            }
                        }
                    })
                    .collect(),
            );
            Ok(())
        })?;
        Ok((names, out))
    }

    /// Every row of a query, for `viv db query`: the column names and the
    /// rows.
    pub fn rows(h: f64, sql: &str, p: &Params) -> Result<(Vec<String>, Vec<Vec<Value>>), String> {
        let s = cstr(sql, "query")?;
        with(h, |c| {
            let st = query(c.db, &s, p, "query")?;
            collect(c.db, &st)
        })
    }
}

#[cfg(target_arch = "wasm32")]
mod browser {
    use super::*;
    fn no<T>() -> Result<T, String> {
        Err(err("the db_* builtins (SQLite) are not available in the browser yet".to_string()))
    }
    pub fn open(_: &str) -> Result<f64, String> { no() }
    pub fn close(_: f64) -> Result<f64, String> { no() }
    pub fn exec(_: f64, _: &str, _: &Params) -> Result<f64, String> { no() }
    pub fn col_real(_: f64, _: &str, _: &Params) -> Result<Vec<f64>, String> { no() }
    pub fn col_int(_: f64, _: &str, _: &Params) -> Result<Vec<i64>, String> { no() }
    pub fn col_text(_: f64, _: &str, _: &Params) -> Result<Vec<String>, String> { no() }
    pub fn stage(_: f64, _: &str, _: Col) -> Result<f64, String> { no() }
    pub fn insert(_: f64, _: &str) -> Result<f64, String> { no() }
}
#[cfg(target_arch = "wasm32")]
pub use browser::*;
