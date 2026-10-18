// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Notes attached to diagnostics for habits carried over from Python (and
//! most other languages): `**` and `range`. Viv has neither, and the plain
//! error ("expected an expression") does not say what to write instead.

/// `x ** y`, which the parser sees as `*` followed by a stray `*`.
pub const POW: &str = "Viv has no `**` operator: write `x ^ y` (or `pow(x, y)`) for x to the power y";

/// An array where a condition goes: `if a == b` on two arrays. A comparison
/// of arrays is an array of 1.0 and 0.0, one per element, not a Bool.
pub const ARRAY_CONDITION: &str = "a comparison of arrays gives an array of 1.0 and 0.0: test it with `all(a == b)` or `any(a > 0)`";

/// `for i in range(n)`.
pub const FOR_IN: &str = "a Viv loop is `for i to n { ... }`, which runs i = 0, 1, ..., n - 1 (Python's `for i in range(n)`)";

/// Names the compiler matches by name only to refuse them with a hint; they
/// are not builtins (the builtin tables' tests skip them).
pub const REFUSED_NAMES: &[&str] = &["range"];

/// A call to `range(...)` anywhere else.
pub const RANGE: &str = "Viv has no `range`: loop with `for i to n { ... }`, which runs i = 0, 1, ..., n - 1";

/// `2x`, from Julia or from math on paper. The lexer used to read it as the
/// number `2` and then a separate statement `x`, so `y = 2x` set y to 2.
pub fn juxtapose(number: &str, name: &str) -> String {
    format!("Viv has no implicit multiplication: write `{number} * {name}`")
}
