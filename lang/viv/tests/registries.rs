// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Arrays of function values -- the REGISTRY that lets a contract hold its
//! strategies as data instead of spelling them out.
//!
//! The interpreter has had boxed arrays for a while; what these tests pin down
//! is the compiled side, where an array is packed and a call is normally
//! specialized away. A registry is read by an index nobody knows until it runs,
//! so there is nothing to specialize: it dispatches through a function pointer.
//! Every test therefore checks the two engines print the same thing.

use knotkit::span::SourceId;

/// Compile `src` to C, build it, run it, return stdout. `None` if there is no C
/// compiler, so the suite still passes on a machine without one.
fn run_compiled(src: &str, tag: &str) -> Option<String> {
    let c = viv::compile_c(src).expect("should compile");
    let dir = std::env::temp_dir().join(format!("viv_reg_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("p.c"), &c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let built = std::process::Command::new(&cc)
        .args(["-std=c99", "-O2"])
        .arg(dir.join("p.c"))
        .arg("-lm")
        .arg("-o")
        .arg(dir.join("p"))
        .output()
        .ok()?;
    assert!(built.status.success(), "generated C did not compile:\n{}", String::from_utf8_lossy(&built.stderr));
    let out = std::process::Command::new(dir.join("p")).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn interp_lines(src: &str) -> String {
    viv::interp::run(src)
        .expect("should interpret")
        .lines()
        .filter(|l| !l.starts_with("__"))
        .map(|l| format!("{l}\n"))
        .collect()
}

fn agree(src: &str, tag: &str) -> String {
    let interp = interp_lines(src);
    if let Some(compiled) = run_compiled(src, tag) {
        assert_eq!(interp, compiled, "the engines disagree");
    }
    interp
}

const SQUARES: &str = "def sq(x) {\n    return x * x\n}\ndef cube(x) {\n    return x * x * x\n}\n";

#[test]
fn a_registry_is_built_indexed_and_measured() {
    let out = agree(
        &format!("{SQUARES}ms = [sq, cube]\nshow len(ms)\nshow apply(ms[0], 3.0)\nshow apply(ms[1], 3.0)\n"),
        "basic",
    );
    assert_eq!(out, "2\n9\n27\n", "{out}");
}

#[test]
fn a_registry_is_dispatched_by_a_value_the_compiler_does_not_know() {
    // The index comes out of a loop, so no specialization could have chosen
    // the callee: this is an indirect call through a function pointer.
    let out = agree(
        &format!("{SQUARES}ms = [sq, cube]\ntotal = 0.0\nfor i to len(ms) {{\n    total = total + apply(ms[i], 2.0)\n}}\nshow total\n"),
        "loop",
    );
    assert_eq!(out, "total: 12\n", "{out}");
}

#[test]
fn a_registry_travels_as_an_argument() {
    let out = agree(
        &format!(
            "{SQUARES}def run_all(ms, x) {{\n    t = 0.0\n    for i to len(ms) {{\n        t = t + apply(ms[i], x)\n    }}\n    return t\n}}\nshow run_all([sq, cube], 2.0)\n"
        ),
        "arg",
    );
    assert_eq!(out, "12\n", "{out}");
}

const STRATEGIES: &str = "def target(x) {\n    return x * x - 2.0\n}\ndef s_left(f, a, b) {\n    return f(a)\n}\ndef s_right(f, a, b) {\n    return f(b)\n}\n";

#[test]
fn a_registry_of_strategies_takes_the_function_it_works_on() {
    // The shape every contract has: each strategy's FIRST argument is itself a
    // function. Specialization cannot serve this, so `s_left` and `s_right`
    // survive as generic copies whose `f` is a runtime function value.
    let out = agree(
        &format!("{STRATEGIES}ms = [s_left, s_right]\nshow apply(ms[0], target, 0.0, 3.0)\nshow apply(ms[1], target, 0.0, 3.0)\n"),
        "strategies",
    );
    assert_eq!(out, "-2\n7\n", "{out}");
}

#[test]
fn a_strategy_registry_and_its_function_both_travel_as_arguments() {
    let out = agree(
        &format!(
            "{STRATEGIES}def worst(ms, f, a, b) {{\n    w = 0.0\n    for i to len(ms) {{\n        w = max(w, fabs(apply(ms[i], f, a, b)))\n    }}\n    return w\n}}\nshow worst([s_left, s_right], target, 0.0, 3.0)\n"
        ),
        "worst",
    );
    assert_eq!(out, "7\n", "{out}");
}

#[test]
fn a_registry_returned_from_a_def_outlives_the_call() {
    // The registry is allocated on the same arena the numeric arrays use, so
    // returning one keeps it the way returning an array does.
    let out = agree(
        &format!("{SQUARES}def methods() {{\n    return [sq, cube]\n}}\nshow apply(methods()[1], 4.0)\n"),
        "returned",
    );
    assert_eq!(out, "64\n", "{out}");
}

#[test]
fn the_c_backend_emits_one_wrapper_per_def_however_often_it_is_named() {
    // Two mentions of the same def as a value hit the same content-addressed
    // node; rooting it twice would emit the wrapper twice and C would refuse.
    let c = viv::compile_c(&format!("{SQUARES}a = [sq, cube]\nb = [sq, cube]\nshow apply(a[0], 2.0) + apply(b[1], 2.0)\n"))
        .expect("should compile");
    // the prototype is emitted too, so count the DEFINITION
    assert_eq!(c.matches("viv_wrap_sq(const viv_arg* a, int n) {").count(), 1, "one wrapper for sq:\n{c}");
    assert!(c.contains("viv_fnvec(2,"), "registry literal:\n{c}");
    assert!(c.contains("viv_fnget("), "registry index:\n{c}");
}

#[test]
fn functions_of_different_arity_cannot_share_a_registry() {
    // A C array has one element type, and the uniform ABI reads each argument
    // slot by the callee's signature -- so a registry's members must agree.
    let err = viv::build_ir("def one(x) { return x }\ndef two(x, y) { return x + y }\nms = [one, two]\nshow len(ms)", SourceId(0))
        .unwrap_err();
    assert!(
        err.iter().any(|d| d.message.contains("the same number of arguments")),
        "expected an arity-mismatch diagnostic, got {err:?}"
    );
}

#[test]
fn numbers_and_functions_cannot_share_a_registry() {
    let err = viv::build_ir("def one(x) { return x }\nms = [one, 3.0]\nshow len(ms)", SourceId(0)).unwrap_err();
    assert!(
        err.iter().any(|d| d.message.contains("not a mixture")),
        "expected a mixed-array diagnostic, got {err:?}"
    );
}

#[test]
fn a_registry_slot_cannot_be_assigned() {
    let err = viv::build_ir("def one(x) { return x }\ndef two(x) { return x + 1.0 }\nms = [one, two]\nms[0] = two\nshow len(ms)", SourceId(0))
        .unwrap_err();
    assert!(
        err.iter().any(|d| d.message.contains("cannot assign into a function array")),
        "expected an assignment diagnostic, got {err:?}"
    );
}

#[test]
fn a_registry_member_can_take_and_return_an_array() {
    // The shape a tensor-valued stage has: each strategy is handed the problem
    // as an array and hands back its answer as one. The argument travels in the
    // union's `arr` member and so does the result.
    let out = agree(
        "def dbl(v) {\n    out = zeros(len(v))\n    for i to len(v) {\n        out[i] = 2.0 * v[i]\n    }\n    return out\n}\n\
         def add(v) {\n    out = zeros(len(v))\n    for i to len(v) {\n        out[i] = v[i] + v[i]\n    }\n    return out\n}\n\
         ms = [dbl, add]\nx = [1.0, 2.0, 3.0]\na = apply(ms[0], x)\nb = apply(ms[1], x)\nshow a[2]\nshow b[2] - a[2]\n",
        "arrays",
    );
    assert_eq!(out, "6\n0\n", "{out}");
}

#[test]
fn a_number_where_an_array_is_expected_is_refused() {
    let err = viv::build_ir(
        "def first(v) {\n    return v[0]\n}\nms = [first]\nshow apply(ms[0], 3.0)",
        SourceId(0),
    )
    .unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("should be an array")), "{err:?}");
}

