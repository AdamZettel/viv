// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The kinds of run-time error.
//!
//! Every place a run can stop names its kind explicitly: the C runtime and
//! the C the lowering writes pass a `VIV_E_*` constant to `viv_stop` (or to
//! the fail helper that ends in it), and the interpreter passes an `ErrKind`
//! to `kinded` where it makes the message. Nothing reads a kind back out of a
//! message's words, so rewording a message cannot change the exception an
//! exported program raises.
//!
//! This enum is the one list. embed.c's `#define VIV_E_*` lines, the
//! exported header's `<NAME>_E_*` enum and the Python wrapper's `_KINDS`
//! table are checked against it by the tests below, so they cannot drift.
//!
//! Which kind a site takes:
//! - `Index`: an index, an axis or a slice bound outside its dimension, or
//!   an array indexed with the wrong number of indices (Python: IndexError);
//! - `Value`: an argument of the right type with a value the operation
//!   cannot take: a shape mismatch, a rank the function does not take, a NaN
//!   where none may be, a matrix that is singular or not symmetric or not
//!   positive definite, a zero step, an empty sequence, a string that is not
//!   a number (ValueError);
//! - `Overflow`: a number or a size past a limit: Int overflow, a value that
//!   does not fit in an Int, an array past the 2147483647-element limit, a
//!   string past Viv's length limit (OverflowError);
//! - `ZeroDiv`: a division or remainder by zero (ZeroDivisionError);
//! - `Recursion`: the recursion limit (crate::recursion; RecursionError);
//! - `Memory`: an allocation that failed (MemoryError);
//! - `Interrupt`: the caller asked the run to stop (KeyboardInterrupt);
//! - `Stop`: the program's own `stop(...)` (VivStop);
//! - `Error`: everything else: a solver that did not converge or failed its
//!   certificate, a missing map key, a file that cannot be read, checkpoint
//!   I/O, a placeholder def called, an internal check (VivError).

use std::cell::RefCell;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ErrKind {
    Error = 1,
    Index = 2,
    Value = 3,
    Overflow = 4,
    ZeroDiv = 5,
    Recursion = 6,
    Memory = 7,
    Interrupt = 8,
    Stop = 9,
}

impl ErrKind {
    pub const ALL: [ErrKind; 9] = [
        ErrKind::Error,
        ErrKind::Index,
        ErrKind::Value,
        ErrKind::Overflow,
        ErrKind::ZeroDiv,
        ErrKind::Recursion,
        ErrKind::Memory,
        ErrKind::Interrupt,
        ErrKind::Stop,
    ];

    /// The number the C runtime returns and the wrapper reads.
    pub fn code(self) -> u8 {
        self as u8
    }

    /// The C constant's suffix: `VIV_E_<suffix>` in embed.c and the lowering,
    /// `<NAME>_E_<suffix>` in an exported header.
    pub fn c_suffix(self) -> &'static str {
        match self {
            ErrKind::Error => "ERROR",
            ErrKind::Index => "INDEX",
            ErrKind::Value => "VALUE",
            ErrKind::Overflow => "OVERFLOW",
            ErrKind::ZeroDiv => "ZERODIV",
            ErrKind::Recursion => "RECURSION",
            ErrKind::Memory => "MEMORY",
            ErrKind::Interrupt => "INTERRUPT",
            ErrKind::Stop => "STOP",
        }
    }

    /// The C constant: `VIV_E_INDEX`.
    pub fn c_name(self) -> String {
        format!("VIV_E_{}", self.c_suffix())
    }

    /// The exception the Python wrapper raises. An interrupt raises Python's
    /// own KeyboardInterrupt, so it has no class of the wrapper's.
    pub fn python_class(self) -> Option<&'static str> {
        Some(match self {
            ErrKind::Error => "VivError",
            ErrKind::Index => "VivIndexError",
            ErrKind::Value => "VivValueError",
            ErrKind::Overflow => "VivOverflowError",
            ErrKind::ZeroDiv => "VivZeroDivisionError",
            ErrKind::Recursion => "VivRecursionError",
            ErrKind::Memory => "VivMemoryError",
            ErrKind::Interrupt => return None,
            ErrKind::Stop => "VivStop",
        })
    }

    pub fn from_code(code: u8) -> Option<ErrKind> {
        ErrKind::ALL.into_iter().find(|k| k.code() == code)
    }

    /// The `#define VIV_E_* n` lines embed.c holds.
    pub fn c_defines() -> String {
        ErrKind::ALL.iter().map(|k| format!("#define {} {}\n", k.c_name(), k.code())).collect()
    }
}

// The interpreter's errors are strings (`Result<_, String>` through the
// tree walk). The site that makes one with a kind other than `Error` records
// the kind here with the message it made; the run's report takes it back
// with `kind_of`, which requires the reported message to contain that one,
// so a kinded error the interpreter made and then dropped does not lend its
// kind to a later one.
thread_local! {
    static LAST: RefCell<Option<(ErrKind, String)>> = const { RefCell::new(None) };
}

/// An interpreter error message with its kind: `Err(kinded(ErrKind::Index, msg))`.
pub fn kinded(kind: ErrKind, msg: impl Into<String>) -> String {
    let msg = msg.into();
    LAST.with(|l| *l.borrow_mut() = Some((kind, msg.clone())));
    msg
}

/// Forget the last kinded error: at the start of a run.
pub fn reset() {
    LAST.with(|l| *l.borrow_mut() = None);
}

/// The kind of the error a run stopped with, given its message: the kind
/// its site recorded, or `Error` for a site that recorded none.
pub fn kind_of(msg: &str) -> ErrKind {
    LAST.with(|l| match &*l.borrow() {
        Some((k, m)) if msg.contains(m.as_str()) => *k,
        _ => ErrKind::Error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ident_char(c: char) -> bool {
        c.is_ascii_alphanumeric() || c == '_'
    }

    #[test]
    fn codes_are_distinct_and_round_trip() {
        for (i, k) in ErrKind::ALL.iter().enumerate() {
            assert_eq!(k.code() as usize, i + 1);
            assert_eq!(ErrKind::from_code(k.code()), Some(*k));
        }
    }

    /// embed.c's constants are this enum's, and it has no other `VIV_E_`.
    #[test]
    fn embed_c_defines_the_same_kinds() {
        let embed = crate::lower::EMBED_C;
        assert!(embed.contains(&ErrKind::c_defines()), "embed.c's #define VIV_E_* block is ErrKind::c_defines():\n{}", ErrKind::c_defines());
        let named: Vec<&str> = embed.lines().filter_map(|l| l.strip_prefix("#define VIV_E_")).collect();
        assert_eq!(named.len(), ErrKind::ALL.len(), "{named:?}");
    }

    /// The exported header's enum and the Python wrapper's table, as
    /// crate::export writes them.
    #[test]
    fn the_export_writes_the_same_kinds() {
        let src = include_str!("export.rs");
        let header: String = ErrKind::ALL
            .iter()
            .map(|k| format!("{{up}}_E_{} = {}", k.c_suffix(), k.code()))
            .collect::<Vec<_>>()
            .join(", ");
        let flat = src.replace(",\\n       ", ", ");
        assert!(flat.contains(&format!("enum {{{{ {header} }}}};")), "export.rs's header enum is {header}");
        let py: String = ErrKind::ALL
            .iter()
            .filter_map(|k| k.python_class().map(|c| format!("{}: {c}", k.code())))
            .collect::<Vec<_>>()
            .join(", ");
        let flat = src.replace(",\n          ", ", ");
        assert!(flat.contains(&format!("_KINDS = {{{py}}}")), "export.rs's _KINDS is {{{py}}}");
        for k in ErrKind::ALL {
            if let Some(c) = k.python_class() {
                if c != "VivError" {
                    assert!(src.contains(&format!("class {c}(VivError")), "the wrapper defines {c}");
                }
            }
        }
    }

    /// Every stop names its kind: each `viv_stop`/`viv_stop_exit` call in the
    /// C runtime and in the C the lowering writes passes a `VIV_E_*` constant
    /// of this enum, or forwards the `kind` parameter of a fail helper; and
    /// every call of such a helper (a C function whose first parameter is
    /// `int kind`) does the same. The kind that used to be read from the
    /// message (`VIV_E_AUTO`, `viv_stop_kind`) is gone.
    #[test]
    fn every_stop_site_names_its_kind() {
        fn walk(d: &std::path::Path, out: &mut Vec<(String, String)>) {
            for e in std::fs::read_dir(d).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() {
                    walk(&p, out);
                } else if matches!(p.extension().and_then(|x| x.to_str()), Some("rs" | "c" | "h")) && !p.ends_with("errkind.rs") {
                    out.push((p.display().to_string(), std::fs::read_to_string(&p).unwrap()));
                }
            }
        }
        let mut texts: Vec<(String, String)> = Vec::new();
        walk(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut texts);
        let kinds: Vec<String> = ErrKind::ALL.iter().map(|k| k.c_name()).collect();
        // the helpers: viv_stop, viv_stop_exit, and every C function whose
        // first parameter is `int kind`
        let mut helpers: Vec<String> = vec!["viv_stop".into(), "viv_stop_exit".into()];
        for (_, t) in &texts {
            for (at, _) in t.match_indices("(int kind") {
                let name: String = t[..at].chars().rev().take_while(|&c| ident_char(c)).collect::<Vec<_>>().into_iter().rev().collect();
                if name.starts_with("viv_") && !helpers.contains(&name) {
                    helpers.push(name);
                }
            }
        }
        assert!(helpers.len() > 8, "the fail helpers were found: {helpers:?}");
        let mut bad = Vec::new();
        let mut sites = 0;
        for (path, t) in &texts {
            assert!(!t.contains("VIV_E_AUTO") && !t.contains("viv_stop_kind"), "{path}: a kind read from the message");
            for h in &helpers {
                let pat = format!("{h}(");
                for (at, _) in t.match_indices(&pat) {
                    if t[..at].chars().next_back().is_some_and(ident_char) {
                        continue;
                    }
                    // a definition, a prototype or a macro, not a call
                    let from = t[..at].rfind(['\n', ';', '{', '}']).map_or(0, |i| i + 1);
                    let before = &t[from..at];
                    if before.contains("void") || before.contains("static") || before.contains("define ") {
                        continue;
                    }
                    let rest = t[at + pat.len()..].trim_start();
                    let arg: String = rest.chars().take_while(|&c| ident_char(c)).collect();
                    sites += 1;
                    if !(kinds.contains(&arg) || arg == "kind") {
                        let line = t[..at].matches('\n').count() + 1;
                        bad.push(format!("{path}:{line}: {h}({arg}"));
                    }
                }
            }
        }
        assert!(bad.is_empty(), "stops without an explicit kind:\n{}", bad.join("\n"));
        assert!(sites > 150, "the scan found the stop sites ({sites})");
    }

    #[test]
    fn an_interpreter_error_keeps_its_kind_only_while_it_is_the_one_reported() {
        reset();
        assert_eq!(kind_of("anything"), ErrKind::Error);
        let m = kinded(ErrKind::Index, "index 3 out of bounds");
        assert_eq!(kind_of(&format!("{m}\n  --> f.viv:1:1")), ErrKind::Index);
        assert_eq!(kind_of("a later error with no kind"), ErrKind::Error);
        reset();
    }
}
