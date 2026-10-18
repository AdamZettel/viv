// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! One program, two languages.
//!
//! A Viv program that `#! use`s a module written in weft. The expander sees the
//! module's `#! lang weft` line and hands it to an injected translator; weft
//! turns the weak form into Viv source; the result is spliced in and compiled
//! like anything else. Viv's own crate knows nothing about weft -- the
//! translator is supplied here, by a driver that can see both.

use knotkit::span::SourceId;

/// The translator a driver supplies: a language name and a module's source, in;
/// Viv source, out.
fn translator(lang: &str, module: &str, src: &str) -> Result<String, String> {
    match lang {
        "weft" => weft::compile_to_viv(src),
        other => Err(format!("module `{module}`: no translator for `{other}`")),
    }
}

#[test]
fn a_viv_program_can_use_a_module_written_in_weft() {
    let form = format!(
        "#! lang weft\n{}",
        std::fs::read_to_string("examples/poisson.weft").expect("the example is there")
    );
    let program = "import forms.poisson\n";

    let mut resolve = |name: &str| -> Result<String, String> {
        if name == "forms/poisson" { Ok(form.clone()) } else { Err(format!("no module {name}")) }
    };
    let expanded = viv::expand_uses_with(program, &mut resolve, &mut |l, m, s| translator(l, m, s))
        .expect("the dialect is translated and spliced");

    // what came back is Viv, not weft: the weak form is gone and a solve is there
    assert!(!expanded.contains("dot(grad(u)"), "the weak form should have been compiled away:\n{expanded}");
    assert!(expanded.contains("def "), "and Viv definitions should have taken its place");

    // and it is a real Viv program, all the way to C
    let ir = viv::build_ir(&expanded, SourceId(0)).expect("the spliced Viv lowers");
    assert!(ir.roots().len() > 0, "it has something to run");
    assert!(viv::compile_c(&expanded).is_ok(), "and it compiles to C");
}

#[test]
fn a_dialect_with_no_translator_says_so_instead_of_guessing() {
    let module = "#! lang gpu\nkernel saxpy(a, x, y) { y = a * x + y }\n";
    let mut resolve = |_: &str| -> Result<String, String> { Ok(module.to_string()) };

    // the plain expander has no translators at all
    let err = viv::expand_uses("import kernels.saxpy\n", &mut resolve).unwrap_err();
    assert!(err.contains("gpu"), "it should name the dialect it cannot read: {err}");

    // and one that has a weft translator still will not pretend to read gpu
    let err = viv::expand_uses_with("import kernels.saxpy\n", &mut resolve, &mut |l, m, s| {
        translator(l, m, s)
    })
    .unwrap_err();
    assert!(err.contains("no translator for `gpu`"), "and say which: {err}");
}

#[test]
fn a_module_with_no_declaration_is_viv_and_nothing_changes() {
    assert_eq!(viv::lang_of("def f(x) {\n    return x\n}\n"), None);
    assert_eq!(viv::lang_of("# a comment first\n#! lang weft\n"), Some("weft".to_string()));
    // a declaration after code is not a declaration
    assert_eq!(viv::lang_of("x = 1.0\n#! lang weft\n"), None);
}
