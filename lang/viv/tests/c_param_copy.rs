// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! An array parameter is copied into a local on entry, with `(void)p;` so a def
//! that never reads it compiles cleanly. The cast must use the parameter's C
//! name, which is escaped when the Viv name is a C keyword or macro (`I`).

#[test]
fn an_unread_array_parameter_named_like_a_c_macro_compiles() {
    let src = "def f(I: Arr) {\n    return 1.0\n}\nprint(f(zeros(2)))\n";
    let c = viv::compile_c(src).expect("compiles to C");
    assert!(c.contains("(void)(I_)"), "the cast names the escaped local");
    assert!(!c.contains("(void)I;"), "no raw, unescaped name");
}
