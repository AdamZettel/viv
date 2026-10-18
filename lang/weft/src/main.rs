// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The Weft CLI: `weft <input.weft> [-o <output>]`.
//!
//! Emits Viv source by default. If the output path ends in `.c`, the generated
//! Viv is lowered the rest of the way (weft -> Viv -> C) in one step.

use std::path::PathBuf;
use std::process::ExitCode;

use knotkit::backend::{Backend, CBackend};
use knotkit::diag::{PlainRenderer, Renderer};
use knotkit::io::{IoEnvironment, NativeIo};
use knotkit::span::SourceMap;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let Some((input, output)) = parse_args(&args[1..]) else {
        eprintln!("usage: weft <input.weft> [-o <output.viv|output.c>]");
        return ExitCode::FAILURE;
    };

    let io = NativeIo;
    let src = match io.read_to_string(&input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", input.display(), e);
            return ExitCode::FAILURE;
        }
    };

    let viv_src = match weft::compile_to_viv(&src) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let out_path = output.unwrap_or_else(|| input.with_extension("viv"));
    let to_c = out_path.extension().map_or(false, |e| e == "c");

    let bytes: Vec<u8> = if to_c {
        // Continue weft -> Viv -> C in one step.
        let mut sources = SourceMap::new();
        let source_id = sources.add(format!("{} (generated)", input.display()), viv_src.clone());
        let program = match viv::build_ir(&viv_src, source_id) {
            Ok(p) => p,
            Err(diags) => {
                let r = PlainRenderer;
                for d in &diags {
                    eprint!("{}", r.render(d, &sources));
                }
                return ExitCode::FAILURE;
            }
        };
        match CBackend.emit(&program, &io) {
            Ok(e) => e.artifact.as_c().expect("C backend produces C").as_bytes().to_vec(),
            Err(diags) => {
                let r = PlainRenderer;
                for d in &diags {
                    eprint!("{}", r.render(d, &sources));
                }
                return ExitCode::FAILURE;
            }
        }
    } else {
        viv_src.into_bytes()
    };

    if let Err(e) = io.write(&out_path, &bytes) {
        eprintln!("error: cannot write {}: {}", out_path.display(), e);
        return ExitCode::FAILURE;
    }
    println!("wrote {}", out_path.display());
    ExitCode::SUCCESS
}

fn parse_args(args: &[String]) -> Option<(PathBuf, Option<PathBuf>)> {
    let mut input = None;
    let mut output = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" => {
                i += 1;
                output = Some(PathBuf::from(args.get(i)?));
            }
            other => {
                if input.is_some() {
                    return None;
                }
                input = Some(PathBuf::from(other));
            }
        }
        i += 1;
    }
    Some((input?, output))
}
