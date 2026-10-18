// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Weft: a UFL-style weak-form DSL on knotkit.
//!
//! A weak form (`bilinear: dot(grad(u), grad(v))`, `linear: f * v`) is parsed into
//! a small symbolic IR, then a form compiler lowers it to **Viv** source — which
//! Viv's existing pipeline turns into faithful C99 / WASM / OpenMP. The compiler
//! only writes the symbolic front end + the element kernel; the entire numerical
//! runtime (arrays, `solve_linear`, the mesh loop) is inherited from Viv.
//!
//! Pipeline:  weft source -> Form IR -> [form compiler] -> Viv source -> C.

pub mod compile;
pub mod expr;
pub mod parse;

/// Compile weft source into a Viv program (source text).
pub fn compile_to_viv(src: &str) -> Result<String, String> {
    let problem = parse::parse_problem(src)?;
    compile::compile(&problem)
}
