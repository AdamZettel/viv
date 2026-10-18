// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! `viv claims`: the claims in claims/ (a statement, its Viv program and
//! its vipyr twin) and their recorded status, through tools/claims/run.py.
//!
//!     viv claims                    list them with their status (claims/results.json)
//!     viv claims run [--all|--heavy|--only <slug>] [--write] [--check]
//!                                     run them (the fast ones by default)
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const USAGE: &str = "usage: viv claims [--json] [--only <slug>]   list the claims and their recorded status
       viv claims run [--all | --heavy | --only <slug>] [--write] [--check]";

/// The repository root: the nearest ancestor of the current directory that
/// holds Cargo.toml and claims/.
fn repo_root() -> Option<PathBuf> {
    let cwd = std::fs::canonicalize(".").ok()?;
    cwd.ancestors().find(|a| a.join("Cargo.toml").exists() && a.join("claims").is_dir()).map(Path::to_path_buf)
}

pub fn cli(args: &[String]) -> ExitCode {
    let rest: Vec<String> = args.iter().skip(2).cloned().collect();
    if rest.first().map(String::as_str) == Some("--help") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let Some(root) = repo_root() else {
        eprintln!("viv claims: run it inside the Viv repository (no claims/ above this directory)\n{USAGE}");
        return ExitCode::from(2);
    };
    let mut cmd = Command::new("python3");
    cmd.arg(root.join("tools/claims/run.py")).current_dir(&root);
    match rest.first().map(String::as_str) {
        None | Some("--json") | Some("--only") => {
            cmd.arg("--list").args(&rest);
        }
        Some("run") => {
            cmd.args(&rest[1..]);
        }
        Some(other) => {
            eprintln!("viv claims: unknown argument `{other}`\n{USAGE}");
            return ExitCode::from(2);
        }
    }
    if std::env::var_os("VIV").is_none() {
        if let Ok(me) = std::env::current_exe() {
            cmd.env("VIV", me);
        }
    }
    match cmd.status() {
        Ok(s) if s.success() => ExitCode::SUCCESS,
        Ok(_) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("viv claims: cannot run python3: {e}");
            ExitCode::FAILURE
        }
    }
}
