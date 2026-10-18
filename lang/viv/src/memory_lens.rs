// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The memory lens file, `<module>.memory` (sections
//! 1.3 and 9): facts about a module's memory that an author pins beside the
//! code, one entry per def, anchored by the def's name and stamped with its
//! body hash like `.intent` and `.debug`.
//!
//! ```text
//! # lib/hf/md_eri.memory
//! mderi_build  @7c9510bf
//!   disjoint  parallel for q: Gp   "pass q writes the canonical slots of the quartet QL[q] ..."
//! ```
//!
//! Stage 1 reads one kind of entry, `disjoint`: the passes of the def's
//! `parallel for <var>` write disjoint elements of the arrays named, for a
//! reason the compiler cannot see. `crate::parallel` accepts the loop's
//! writes to them, and every checked build tests the pin while it runs: each
//! element a pass writes is recorded, and a second pass writing it stops the
//! run naming both passes (the C runtime's `viv_dj_*`, the interpreter's
//! `Interp::dj_note`, the wasm runtime's `vrt_dj_*`). A release build trusts
//! it. The stamp is read and kept; `viv lens check` does not yet report a
//! stale `.memory` entry (that comes with the lens of stage 2), and a stale
//! pin is still applied and tested, as the design says of stale pins.
//! The other entries of the design (`bounded per iteration`, `no copy`,
//! `in place`, `no checks`) come with the ownership pass (stage 2) and are
//! refused here, so that no one writes a pin nothing checks.
//!
//! Where the pins come from. A driver that reads files (the CLI) hands over
//! the lens files beside the program, its `--use` preludes and its `#! use`
//! modules (`set_loaded`). A program assembled elsewhere (the site expands
//! `#! use` lines in the browser, the build scripts in node) reaches the
//! compiler as text, so the compiler also carries the repository's own lens
//! files (`REPO`). A pin names its def and its loop variable, and applies to
//! that loop wherever the def is, so a module's pins hold whether it is the
//! program or one of its modules. (A program of its own that reused both
//! names would get the pin too; the pin is still tested while it runs.)

use std::sync::Mutex;

use knotkit::diag::Diagnostic;

use crate::parallel::Pin;

/// The repository's lens files, by module: what a program assembled away from
/// the files (the site, the node build scripts) is compiled with.
/// `tests/parallel_races.rs` checks that every `.memory` file in the
/// repository is listed, as it is on disk.
pub const REPO: &[(&str, &str, &str)] = &[
    ("lib/hf/md_eri", "lib/hf/md_eri.memory", include_str!("../../../lib/hf/md_eri.memory")),
    ("lib/hf/mdg_eri", "lib/hf/mdg_eri.memory", include_str!("../../../lib/hf/mdg_eri.memory")),
];

/// The lens files a driver read for the program it is compiling: (file name
/// for messages, text). Process-wide, since the CLI runs the interpreter on a
/// thread of its own.
static LOADED: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());

/// The lens files a driver read for the next programs it compiles (the
/// program's own, its preludes', its modules'). They are used instead of the
/// carried copies.
pub fn set_loaded(files: Vec<(String, String)>) {
    *LOADED.lock().unwrap_or_else(|e| e.into_inner()) = files;
}

/// One entry of a lens file: a def and its pins.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub def: String,
    /// the body hash it was stamped with, when it has one
    pub hash: Option<String>,
    pub disjoint: Vec<(String, Vec<String>, String)>,
}

/// Read a lens file. `file` names it in errors.
pub fn parse_file(text: &str, file: &str) -> Result<Vec<Entry>, String> {
    let mut out: Vec<Entry> = Vec::new();
    for (k, raw) in text.lines().enumerate() {
        let line = raw.trim_end();
        let at = || format!("{file}:{}", k + 1);
        let t = line.trim_start();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if !line.starts_with(' ') && !line.starts_with('\t') {
            let mut words = t.split_whitespace();
            let def = words.next().unwrap_or_default().to_string();
            let hash = match words.next() {
                Some(h) => Some(h.strip_prefix('@').ok_or_else(|| format!("{}: after the def's name comes its stamp, `@<hash>`, not `{h}`", at()))?.to_string()),
                None => None,
            };
            if let Some(extra) = words.next() {
                return Err(format!("{}: one def per entry line, then its pins indented below it; `{extra}` is left over", at()));
            }
            out.push(Entry { def, hash, disjoint: Vec::new() });
            continue;
        }
        let Some(entry) = out.last_mut() else {
            return Err(format!("{}: a pin comes after the line naming its def", at()));
        };
        // the reason, a quoted sentence at the end
        let (fact, reason) = match t.find('"') {
            Some(q) => {
                let r = &t[q..];
                if !(r.len() >= 2 && r.ends_with('"')) {
                    return Err(format!("{}: the reason is one quoted sentence at the end of the line", at()));
                }
                (t[..q].trim_end(), r[1..r.len() - 1].to_string())
            }
            None => (t, String::new()),
        };
        let words: Vec<&str> = fact.split_whitespace().collect();
        match words.as_slice() {
            ["disjoint", "parallel", "for", rest @ ..] if !rest.is_empty() => {
                let rest = rest.join(" ");
                let Some((var, arrays)) = rest.split_once(':') else {
                    return Err(format!("{}: `disjoint parallel for <var>: <arrays>`, e.g. `disjoint parallel for q: Gp`", at()));
                };
                let arrays: Vec<String> = arrays.split(',').map(|a| a.trim().to_string()).filter(|a| !a.is_empty()).collect();
                if var.trim().is_empty() || arrays.is_empty() {
                    return Err(format!("{}: `disjoint parallel for <var>: <arrays>` names the loop variable and at least one array", at()));
                }
                if reason.is_empty() {
                    return Err(format!("{}: a `disjoint` pin says why the passes write disjoint elements, as a quoted sentence; the compiler cannot see it, so the reader must", at()));
                }
                entry.disjoint.push((var.trim().to_string(), arrays, reason));
            }
            [first, ..] if ["bounded", "no", "in"].contains(first) => {
                return Err(format!("{}: `{fact}` is a pin the ownership pass will check; until then only `disjoint` entries are read", at()));
            }
            _ => return Err(format!("{}: not a memory pin: `{fact}` (a pin is `disjoint parallel for <var>: <arrays> \"why\"`)", at())),
        }
    }
    Ok(out)
}

/// The pins of a lens file, for `crate::parallel`.
fn pins_of(text: &str, file: &str) -> Result<Vec<Pin>, String> {
    let mut out = Vec::new();
    for e in parse_file(text, file)? {
        for (var, arrays, _) in e.disjoint {
            out.push(Pin { def: e.def.clone(), var, arrays, file: file.to_string() });
        }
    }
    Ok(out)
}

/// The pins that apply to a program: the lens files the driver read, and the
/// lens files the compiler carries, except those the driver read from disk
/// (the file on disk is the newer word). A pin names its def and loop
/// variable, so it applies only to a loop of that name in a def of that name.
pub fn pins_for(_src: &str) -> Result<Vec<Pin>, Vec<Diagnostic>> {
    let err = |e: String| vec![Diagnostic::error(e)];
    let loaded = LOADED.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let mut out = Vec::new();
    for (file, text) in &loaded {
        out.extend(pins_of(text, file).map_err(err)?);
    }
    for (_, file, text) in REPO {
        if loaded.iter().any(|(f, _)| f.ends_with(file)) {
            continue;
        }
        out.extend(pins_of(text, file).map_err(err)?);
    }
    Ok(out)
}
