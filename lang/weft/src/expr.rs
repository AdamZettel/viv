// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The shared expression IR for `weft`. It carries both *form* expressions (over
//! the trial/test functions `u`/`v`, `grad`, `dot`) and *scalar* expressions (over
//! the spatial coordinates `x`/`y` and named coefficients). The form compiler
//! interprets `Var("u")`/`Var("v")` as the trial/test arguments.

#[derive(Clone, Debug, PartialEq)]
pub enum E {
    Num(f64),
    /// An identifier: `u`/`v` (in forms), `x`/`y` (in scalar fns), or a coefficient.
    Var(String),
    /// `grad(e)`.
    Grad(Box<E>),
    /// `dot(a, b)`.
    Dot(Box<E>, Box<E>),
    /// Unary minus.
    Neg(Box<E>),
    /// A binary arithmetic operator: `+ - * /`.
    Bin(char, Box<E>, Box<E>),
    /// A passthrough call to a Viv builtin (`exp`/`sqrt`/`pow`).
    Call(String, Vec<E>),
}
