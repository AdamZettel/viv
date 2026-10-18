// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! `viv facts` and the `vivhash1` def identity:
//! whitespace, comments, docs, step headings and the def's own name leave a
//! def's `ast_hash` alone; any change to its code moves it.

use viv::facts::ast_hashes;

const BASE: &str = "def scale(xs: Arr, k: Num) -> Arr {
    out = zeros(len(xs))
    for i to len(xs) {
        out[i] = k * xs[i]
    }
    return out
}

def total(xs: Arr) -> Num {
    return sum(scale(xs, 2.0))
}
";

fn hash_of(src: &str, name: &str) -> String {
    ast_hashes(src).unwrap().into_iter().find(|(n, _)| n == name).unwrap_or_else(|| panic!("no def {name}")).1
}

#[test]
fn the_hash_is_full_length_hex() {
    let h = hash_of(BASE, "scale");
    assert_eq!(h.len(), 64);
    assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn whitespace_and_blank_lines_do_not_move_it() {
    let spaced = BASE.replace("out = zeros(len(xs))", "out   =   zeros( len(xs) )\n\n").replace("    return out", "\n        return out");
    assert_eq!(hash_of(BASE, "scale"), hash_of(&spaced, "scale"));
}

#[test]
fn comments_docs_and_step_headings_do_not_move_it() {
    let noted = BASE
        .replace("def scale(", "#| what: multiplies every element by k\n# a plain comment\ndef scale(")
        .replace("    out = zeros(len(xs))", "    #: allocate\n    out = zeros(len(xs))  # the result\n    #: fill");
    assert_eq!(hash_of(BASE, "scale"), hash_of(&noted, "scale"));
}

#[test]
fn renaming_the_def_does_not_move_it_but_its_callers_move() {
    let renamed = BASE.replace("def scale(", "def times(").replace("sum(scale(", "sum(times(");
    assert_eq!(hash_of(BASE, "scale"), hash_of(&renamed, "times"));
    // the caller's text changed (it names a different callee), so it moves
    assert_ne!(hash_of(BASE, "total"), hash_of(&renamed, "total"));
}

#[test]
fn body_changes_move_it() {
    let b = hash_of(BASE, "scale");
    for edited in [
        BASE.replace("k * xs[i]", "xs[i] * k"),
        BASE.replace("k * xs[i]", "k + xs[i]"),
        BASE.replace("2.0", "3.0"),
        BASE.replace("k: Num) -> Arr", "k: Num)"),
        BASE.replace("(xs: Arr, k: Num)", "(xs: Arr, c: Num)").replace("k * xs[i]", "c * xs[i]"),
    ] {
        let name = if edited.contains("3.0") { "total" } else { "scale" };
        let before = if name == "total" { hash_of(BASE, "total") } else { b.clone() };
        assert_ne!(before, hash_of(&edited, name), "{edited}");
    }
}

#[test]
fn a_callee_body_change_leaves_the_caller_alone() {
    let edited = BASE.replace("k * xs[i]", "k * xs[i] + 0.0");
    assert_ne!(hash_of(BASE, "scale"), hash_of(&edited, "scale"));
    assert_eq!(hash_of(BASE, "total"), hash_of(&edited, "total"));
}
