// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The inputs a run reads, for the gate's cache (tools/gate/cache.py).
//!
//! With `VIV_TRACE_INPUTS=<file>` in the environment, every file the
//! compiler or the interpreter looks for or reads is appended to `<file>`,
//! one line each:
//!
//! ```text
//!     f <TAB> <absolute path> <TAB> 1|0     a file read (1), or looked for and absent (0)
//!     d <TAB> <absolute path>               a whole directory is an input (py2viv's sources)
//!     io <TAB> <absolute path>              this program's text names read_text, read_lines
//!                                           or program_graph: what it reads at run time is
//!                                           traced only when the interpreter runs it
//! ```
//!
//! A module looked for and absent is an input too: a file created there
//! later would be found first. The gate keys a cached verdict on the
//! contents (and the absence) of exactly the files listed, so a change to
//! any of them reruns the check. Lines are appended with one write each
//! (O_APPEND), so concurrent runs may share a file. Without the variable,
//! nothing is written and the cost is one environment lookup per process.

use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
fn sink() -> Option<&'static PathBuf> {
    use std::sync::OnceLock;
    static SINK: OnceLock<Option<PathBuf>> = OnceLock::new();
    SINK.get_or_init(|| std::env::var_os("VIV_TRACE_INPUTS").filter(|v| !v.is_empty()).map(PathBuf::from)).as_ref()
}

#[cfg(not(target_arch = "wasm32"))]
fn absolute(p: &Path) -> PathBuf {
    if let Ok(c) = std::fs::canonicalize(p) {
        return c;
    }
    let a = if p.is_absolute() { p.to_path_buf() } else { std::env::current_dir().map(|d| d.join(p)).unwrap_or_else(|_| p.to_path_buf()) };
    // an absent file: canonicalize its directory, so the path matches the one a present file gets
    match (a.parent().and_then(|d| std::fs::canonicalize(d).ok()), a.file_name()) {
        (Some(d), Some(n)) => d.join(n),
        _ => a,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn emit(line: String) {
    use std::io::Write;
    let Some(s) = sink() else { return };
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(s) {
        let _ = f.write_all(line.as_bytes());
    }
}

/// A file read (`exists`), or looked for and absent.
pub fn note(path: &Path, exists: bool) {
    #[cfg(not(target_arch = "wasm32"))]
    if sink().is_some() {
        emit(format!("f\t{}\t{}\n", absolute(path).display(), if exists { 1 } else { 0 }));
    }
    #[cfg(target_arch = "wasm32")]
    let _ = (path, exists);
}

/// `path.exists()`, noted.
pub fn exists(path: &Path) -> bool {
    let e = path.exists();
    note(path, e);
    e
}

/// Every file under `dir` is an input.
pub fn note_dir(dir: &Path) {
    #[cfg(not(target_arch = "wasm32"))]
    if sink().is_some() {
        emit(format!("d\t{}\n", absolute(dir).display()));
    }
    #[cfg(target_arch = "wasm32")]
    let _ = dir;
}

/// The builtins that read files named at run time.
pub const RUNTIME_READS: &[&str] = &["read_text", "read_lines", "read_bytes", "program_graph"];

/// Note `program` as one that reads files at run time, when its loaded text
/// names one of RUNTIME_READS as a word (in code, a string or a comment
/// alike: a mention is enough to be careful).
pub fn note_runtime_reads(program: &Path, text: &str) {
    #[cfg(not(target_arch = "wasm32"))]
    if sink().is_some() && names_runtime_read(text) {
        emit(format!("io\t{}\n", absolute(program).display()));
    }
    #[cfg(target_arch = "wasm32")]
    let _ = (program, text);
}

/// Whether `text` has one of RUNTIME_READS as a whole word.
pub fn names_runtime_read(text: &str) -> bool {
    let word = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let b = text.as_bytes();
    RUNTIME_READS.iter().any(|w| {
        text.match_indices(w).any(|(i, _)| {
            let j = i + w.len();
            (i == 0 || !word(b[i - 1])) && (j >= b.len() || !word(b[j]))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_reads_are_whole_words() {
        assert!(names_runtime_read("x = read_text(\"a.txt\")"));
        assert!(names_runtime_read("g = program_graph(\"m\")"));
        assert!(!names_runtime_read("my_read_text_helper(1)"));
        assert!(!names_runtime_read("read_texts = 1"));
    }
}
