// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! knotkit: a kit for building content-addressed, ownership-aware, multi-target
//! compilers. It is built to honor the knotkit commitments, the kit's
//! architectural charter (a separate document, not in this repository; the
//! design documents cite it by section).
//!
//! The kit owns *mechanism* (structural identity, diagnostics, the backend
//! traversal); a language owns *vocabulary* (its node variants and their rules).
//! Nothing Knot-specific belongs in this crate.

pub mod span;
pub mod ir;
pub mod diag;
pub mod io;
pub mod backend;
pub mod print;
