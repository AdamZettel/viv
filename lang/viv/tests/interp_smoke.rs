// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Smoke test: the interpreter runs real example programs and matches expectations.
#[test]
fn area_and_control() {
    let area = std::fs::read_to_string("examples/area.viv").unwrap();
    let out = viv::interp::run(&area).unwrap();
    println!("--- area.viv ---\n{out}");
    assert!(out.contains("12.5664"));  // area(2.0) = pi*4
    assert!(out.contains("42"));       // twice(21)

    let control = std::fs::read_to_string("examples/control.viv").unwrap();
    let out = viv::interp::run(&control).unwrap();
    println!("--- control.viv ---\n{out}");
    assert!(out.contains("120")); // fact(5)
    assert!(out.contains("10"));  // sum_range(5) = 0+1+2+3+4
}

#[test]
fn arrays_and_loops() {
    let src = r#"
a = zeros(3)
for i to 3 { a[i] = i * i }
show a
s = 0.0
for i to len(a) { s = s + a[i] }
print(s)
"#;
    let out = viv::interp::run(src).unwrap();
    println!("--- arrays ---\n{out}");
    assert!(out.contains("[0, 1, 4]"));
    assert!(out.contains("5"));  // 0+1+4
}

// Regression: a name used as a top-level `for` variable must remain an
// ordinary global afterward, so a later assignment to it is what reads see.
// The bug: the loop variable was bound into the local scope while top-level
// assignments went to globals, and reads checked scope first — so `i = 3.0`
// after `for i` was silently ignored (the interpreter kept the stale loop
// value). This diverged from the C and wasm backends. See the 18O shell model,
// where a pair-building loop uses `i`/`j` and a later loop reassigns them.
#[test]
fn top_level_loop_var_is_a_normal_global_afterward() {
    let src = r#"
arr = [10.0, 20.0, 30.0, 40.0]
for i to 3 { }
i = 3.0
show arr[i]
"#;
    let out = viv::interp::run(src).unwrap();
    println!("--- loop-var reuse ---\n{out}");
    assert!(out.contains("40"), "reassigned loop var must index arr[3]=40, got:\n{out}");

    // The shell-model shape: reuse `i`/`j` from one loop as plain vars in the
    // next. Count pairs (i<j) of 4 items whose values sum to zero -> exactly 2.
    let src2 = r#"
val = [-2.0, -1.0, 1.0, 2.0]
pi = zeros(6)
pj = zeros(6)
pc = 0.0
for i to 4 { for j to 4 { if i < j { pi[pc] = i; pj[pc] = j; pc = pc + 1.0 } } }
bc = 0.0
for p to 6 { i = pi[p]; j = pj[p]; if val[i] + val[j] == 0.0 { bc = bc + 1.0 } }
print(bc)
"#;
    let out2 = viv::interp::run(src2).unwrap();
    println!("--- pair count ---\n{out2}");
    assert_eq!(out2.lines().last().unwrap().trim(), "2");
}

// Lambdas + the higher-order numeric builtins (deriv/simpson/trapezoid).
// Every expected value is the C backend's own output for the same source
// (the oracle), so the interpreter matches the compiled program to %g.
#[test]
fn lambdas_and_quadrature() {
    let src = r#"
print(simpson(fn(x) -> exp(0.0 - x*x), 0.0, 1.0, 100))
print(trapezoid(fn(x) -> exp(0.0 - x*x), 0.0, 1.0, 100))
print(deriv(fn(x) -> x*x*x, 2.0))
print(deriv(fn(x) -> exp(x), 1.0))
print(deriv(fn(x) -> sqrt(x), 4.0))
print(deriv(fn(x) -> pow(x, 3.0), 2.0))
a = 5.0
print(deriv(fn(x) -> a*x*x, 3.0))
print(simpson(fn(x) -> x*x, 0.0, 2.0, 7))
"#;
    let out = viv::interp::run(src).unwrap();
    println!("--- lambdas ---\n{out}");
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines, vec![
        "0.746824", // simpson of e^{-x^2} on [0,1]
        "0.746818", // trapezoid of the same
        "12",       // d/dx x^3 at 2 = 3*4
        "2.71828",  // d/dx e^x at 1 = e
        "0.25",     // d/dx sqrt(x) at 4 = 1/(2*2)
        "12",       // d/dx x^3 at 2 via pow
        "30",       // d/dx (a x^2) at 3 = 2*a*3 = 30 (closure capture of a)
        "2.66667",  // simpson of x^2 on [0,2], odd n bumped to even
    ]);
}

// A closure stored in a variable and applied like an ordinary function —
// more lenient than the C backend, which only takes literal lambdas.
#[test]
fn closure_valued_variable() {
    let src = r#"
sq = fn(x) -> x * x
print(sq(6.0))
add = fn(x) -> x + 10.0
print(add(sq(3.0)))
"#;
    let out = viv::interp::run(src).unwrap();
    println!("--- closure var ---\n{out}");
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines, vec!["36", "19"]);
}

// The linear-algebra trio added for the SCF notebook: `@`, `transpose`,
// `eig_sym` — checked against hand values and the eigen-reconstruction
// identity A = V diag(w) V^T that only holds if all three agree.
#[test]
fn matmul_transpose_eig_sym() {
    let src = r#"
A = zeros(2, 2)
A[0, 0] = 2.0
A[0, 1] = 1.0
A[1, 0] = 1.0
A[1, 1] = 3.0
w = zeros(2)
V = zeros(2, 2)
eig_sym(A, w, V)
print(w[0])
print(w[1])
D = zeros(2, 2)
D[0, 0] = w[0]
D[1, 1] = w[1]
R = V @ D @ transpose(V)
err = abs(R[0, 0] - A[0, 0]) + abs(R[0, 1] - A[0, 1]) + abs(R[1, 1] - A[1, 1])
assert_near(err, 0.0, 1.0e-9)
"#;
    let out = viv::interp::run(src).unwrap();
    println!("--- eig trio ---\n{out}");
    // eigenvalues of [[2,1],[1,3]] are (5 ∓ sqrt(5))/2 ≈ 1.38197, 3.61803 (ascending)
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines[0], "1.38197");
    assert_eq!(lines[1], "3.61803");
    assert!(out.contains("assert_near PASS"));
}

// linspace: endpoints exact, spacing uniform — and the C backend accepts the
// same program (parity is the contract).
#[test]
fn linspace_builtin() {
    let src = r#"
g = linspace(0.0, 2.0, 5)
print(g[0])
print(g[2])
print(g[4])
print(len(g))
"#;
    let out = viv::interp::run(src).unwrap();
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines, vec!["0", "1", "2", "5"]);
}

#[test]
fn libm_extended_set_matches_reference_values() {
    // print uses %g (6 sig figs), so print the DIFFERENCE from the reference
    // value and assert it is tiny — that tests full double precision through
    // the 6-digit formatter.
    let out = viv::interp::run(concat!(
        "print(erf(1.0) - 0.8427007929497149)\n",
        "print(erf(0.5) - 0.5204998778130465)\n",
        "print(erf(2.0) - 0.9953222650189527)\n",
        "print(erf(-1.0) + 0.8427007929497149)\n",
        "print(gamma(5.0) - 24.0)\n",
        "print(gamma(0.5) - 1.7724538509055159)\n",
        "print(atan2(1.0, 1.0) - 0.7853981633974483)\n",
        "print(mod(7.0, 3.0) - 1.0)\n",
        "print(tanh(1.0) - 0.7615941559557649)\n",
        "print(round(2.5) - 3.0)\n",
    ))
    .expect("run");
    for line in out.lines().filter(|l| !l.starts_with("__")) {
        let v: f64 = line.parse().expect(line);
        assert!(v.abs() < 1e-12, "difference too large: {line}");
    }
}

#[test]
fn deriv_knows_trig_and_log() {
    let out = viv::interp::run(concat!(
        "print(deriv(fn(x) -> sin(x) * x, 2.0) - (sin(2.0) + 2.0 * cos(2.0)))\n",
        "print(deriv(fn(x) -> log(x), 3.0) - 1.0 / 3.0)\n",
        "print(deriv(fn(x) -> cos(x) * cos(x), 1.0) + 2.0 * cos(1.0) * sin(1.0))\n",
    ))
    .expect("run");
    for line in out.lines().filter(|l| !l.starts_with("__")) {
        let v: f64 = line.parse().expect(line);
        assert!(v.abs() < 1e-14, "difference too large: {line}");
    }
}

#[test]
fn string_values_and_docstrings() {
    // a string is a value
    let out = viv::interp::run("x = \"hi\"\nprint(x)").expect("run");
    assert!(out.starts_with("hi\n"), "{out}");
    // and a docstring at the top of a def stays legal and silent.
    let out = viv::interp::run("def f(x) {\n    \"doubles\"\n    return 2.0 * x\n}\nprint(f(4.0))").expect("run");
    assert_eq!(out.lines().next().unwrap().trim(), "8");
}

#[test]
fn matmul_vector_forms() {
    let out = viv::interp::run(concat!(
        "A = [[0.0, 1.0], [1.0, 0.0]]\n",
        "v = [3.0, 7.0]\n",
        "show A @ v\n",
        "show v @ A\n",
        "M = [[1.0, 2.0], [3.0, 4.0]]\n",
        "show M @ [1.0, 1.0]\n",
    ))
    .expect("run");
    let lines: Vec<&str> = out.lines().filter(|l| !l.starts_with("__")).collect();
    assert_eq!(lines, ["[7, 3]", "[7, 3]", "[3, 7]"], "{out}");
}
