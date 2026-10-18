// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The Skein CLI: `skein <input.scm> [-o <output.c>]`. A thin driver over the kit
//! (build IR -> CBackend -> write C), mirroring the Knot/oa CLIs.

use std::path::PathBuf;
use std::process::ExitCode;

use knotkit::backend::{Backend, CBackend};
use knotkit::diag::{PlainRenderer, Renderer};
use knotkit::io::{IoEnvironment, NativeIo};
use knotkit::span::SourceMap;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" => {
                i += 1;
                match args.get(i) {
                    Some(p) => output = Some(PathBuf::from(p)),
                    None => return usage(),
                }
            }
            other if input.is_none() => input = Some(PathBuf::from(other)),
            _ => return usage(),
        }
        i += 1;
    }
    let Some(input) = input else { return usage() };

    let io = NativeIo;
    let src = match io.read_to_string(&input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", input.display(), e);
            return ExitCode::FAILURE;
        }
    };

    let mut sources = SourceMap::new();
    let source_id = sources.add(input.display().to_string(), src.clone());

    let program = match skein::build_ir(&src, source_id) {
        Ok(p) => p,
        Err(diags) => return report(&diags, &sources),
    };
    let emission = match CBackend.emit(&program, &io) {
        Ok(e) => e,
        Err(diags) => return report(&diags, &sources),
    };
    let Some(c) = emission.artifact.as_c() else {
        eprintln!("error: C backend did not produce C");
        return ExitCode::FAILURE;
    };

    let out_path = output.unwrap_or_else(|| input.with_extension("c"));
    if let Err(e) = io.write(&out_path, c.as_bytes()) {
        eprintln!("error: cannot write {}: {}", out_path.display(), e);
        return ExitCode::FAILURE;
    }
    println!("wrote {} ({} IR nodes)", out_path.display(), program.len());
    ExitCode::SUCCESS
}

fn report(diags: &[knotkit::diag::Diagnostic], sources: &SourceMap) -> ExitCode {
    let renderer = PlainRenderer;
    for d in diags {
        eprint!("{}", renderer.render(d, sources));
    }
    ExitCode::FAILURE
}

fn usage() -> ExitCode {
    eprintln!("usage: skein <input.scm> [-o <output.c>]");
    ExitCode::FAILURE
}
