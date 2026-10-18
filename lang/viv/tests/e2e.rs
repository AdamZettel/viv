// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! End-to-end tests for Viv: source -> IR -> C (real `def`/`return`/`print`
//! surface, double-only).

use knotkit::backend::{Backend, CBackend};
use knotkit::io::MemoryIo;
use knotkit::span::SourceId;

fn emit_c(src: &str) -> String {
    let program = viv::build_ir(src, SourceId(0)).expect("build_ir should succeed");
    let io = MemoryIo::new();
    let emission = CBackend.emit(&program, &io).expect("C emission should succeed");
    emission.artifact.as_c().expect("should be C").to_string()
}

#[test]
fn emits_faithful_c() {
    let c = emit_c("def area(r) { return 3.14159 * r * r }");
    assert!(c.contains("double area(double r)"), "signature:\n{c}");
    // Structure is preserved faithfully (left-associative, fully parenthesized).
    assert!(c.contains("return ((3.14159 * r) * r);"), "body:\n{c}");
}

#[test]
fn emits_record_struct_and_access() {
    // A `record` becomes a C struct typedef; a literal is a designated compound
    // literal; field access is plain `.`; field order follows the declaration.
    let c = emit_c("record Vec3 { x: Num, y: Num }\np = Vec3 { y: 2.0, x: 1.0 }\nshow p.x");
    assert!(c.contains("typedef struct {"), "typedef missing:\n{c}");
    assert!(c.contains("} viv_rec_Vec3;"), "struct name missing:\n{c}");
    assert!(c.contains("viv_rec_Vec3 p;"), "global decl missing:\n{c}");
    // Declaration order (x, y), not literal order (y, x).
    assert!(c.contains("((viv_rec_Vec3){ .x = 1.0, .y = 2.0 })"), "compound literal:\n{c}");
    assert!(c.contains("p.x"), "field access missing:\n{c}");
}

#[test]
fn emits_rank3_array_access() {
    // 3D arrays (e.g. a DIIS history) allocate, index, and assign via the
    // rank-3 runtime helpers -- previously an `unsupported index rank 3` panic.
    // Whole-number indices are integers (lower/intidx.rs): the integer
    // accessors, with the mask of dimensions proved in bounds (none here).
    let c = emit_c("t = zeros(2, 2, 2)\nt[1, 0, 1] = 5.0\nshow t[1, 0, 1]");
    assert!(c.contains("viv_zeros3("), "zeros3 missing:\n{c}");
    assert!(c.contains("viv_lset3(t, 1LL, 0LL, 1LL, 5.0, 0)"), "set3 missing:\n{c}");
    assert!(c.contains("viv_lget3(t, 1LL, 0LL, 1LL, 0)"), "get3 missing:\n{c}");
    // Row-major stride: index ((i*d1+j)*d2+k), each index checked first.
    assert!(c.contains("[(ii * a.dims[1] + jj) * a.dims[2] + kk]"), "rank-3 stride missing:\n{c}");
}

#[test]
fn emits_calls_and_unary() {
    let c = emit_c("def f(x) { return g(x) }\ndef g(y) { return -y }");
    assert!(c.contains("return g(x);"), "{c}");
    assert!(c.contains("return -(y);"), "{c}");
}

#[test]
fn print_lowers_to_runtime_and_main() {
    let c = emit_c("def id(x) { return x }\nprint(id(2.0))");
    assert!(c.contains("static void viv_print_num"), "prelude missing:\n{c}");
    assert!(c.contains("int main(void)"), "main missing:\n{c}");
    assert!(c.contains("viv_print_num(id(2.0));"), "print not lowered:\n{c}");
}

#[test]
fn locals_hoisted_and_assigned() {
    // Locals are hoisted (declared zero-init at entry); every assignment is a
    // plain reassignment.
    let c = emit_c("def f(x) {\n  y = x + 1.0\n  y = y * 2.0\n  return y\n}");
    assert!(c.contains("double y = 0;"), "hoisted decl missing:\n{c}");
    assert!(c.contains("y = (x + 1.0);"), "first assign missing:\n{c}");
    assert!(c.contains("y = (y * 2.0);"), "reassign missing:\n{c}");
}

#[test]
fn rejects_unknown_variable() {
    let err = viv::build_ir("def bad(n) { return m }", SourceId(0)).unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("unknown variable `m`")));
}

#[test]
fn rejects_return_at_top_level() {
    let err = viv::build_ir("return 5.0", SourceId(0)).unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("`return` outside")));
}

#[test]
fn rejects_arity_mismatch() {
    let err = viv::build_ir("def f(x) { return x }\nprint(f(1.0, 2.0))", SourceId(0)).unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("expects 1 argument")));
}

#[test]
fn emits_control_flow() {
    let c = emit_c(
        "def f(n) {\n  s = 0.0\n  if n > 0.0 { s = 1.0 } else { s = 2.0 }\n  while s < n { s = s + 1.0 }\n  for i to n { s = s + i }\n  return s\n}",
    );
    assert!(c.contains("if (n > 0.0) {"), "if:\n{c}");
    assert!(c.contains("} else {"), "else:\n{c}");
    assert!(c.contains("while (s < n) {"), "while:\n{c}");
    // the bound read once, an integer counter of the loop's own, the
    // variable set from it (lower/intidx.rs)
    assert!(c.contains("double viv_end_i = n;\n"), "for:\n{c}");
    assert!(c.contains("long long viv_n_i = viv_trip(viv_end_i);\n"), "for:\n{c}");
    assert!(c.contains("for (long long viv_k_i = 0; viv_k_i < viv_n_i; viv_k_i = (viv_k_i + 1)) {\n"), "for:\n{c}");
    assert!(c.contains("i = (double)(viv_k_i);\n"), "for:\n{c}");
}

#[test]
fn hoists_function_scoped_locals() {
    // `y` is bound only inside branches but used after — must be hoisted to the
    // top of the function so it outlives the branch (Viv function scoping).
    let c = emit_c("def g(x) {\n  if x > 0.0 { y = 1.0 } else { y = 2.0 }\n  return y\n}");
    assert!(c.contains("double y = 0;"), "y not hoisted:\n{c}");
    assert!(c.contains("return y;"), "{c}");
}

#[test]
fn arrays_emit_unified_runtime_on_demand() {
    let c = emit_c("v = [1.0, 2.0, 3.0]\nprint(v[0])");
    assert!(c.contains("} viv_arr;"), "unified array type missing:\n{c}");
    assert!(c.contains("viv_vec(3,"), "vec literal not lowered:\n{c}");
    assert!(c.contains("viv_lget1(v, 0LL, 0)"), "index not lowered:\n{c}");
    // On-demand: a vector program must not pull matrix/rank-4 runtime.
    assert!(!c.contains("viv_get4") && !c.contains("viv_lget4"), "rank-4 runtime should be pruned:\n{c}");
    assert!(!c.contains("viv_mat("), "matrix runtime should be pruned:\n{c}");
}

#[test]
fn array_local_in_function_hoisted() {
    // A function-local array is hoisted (declared, zero-initialized) at entry.
    let c = emit_c("def f() {\n  z = zeros(2, 2)\n  z[0, 0] = 1.0\n  return z[0, 0]\n}\nprint(f())");
    assert!(c.contains("viv_arr z = {0};"), "array local decl wrong:\n{c}");
}

#[test]
fn top_level_arrays_are_file_scope_globals() {
    // Top-level variables become file-scope globals (readable in functions).
    let c = emit_c("z = zeros(2, 2)\ndef g() { return z[0, 0] }\nprint(g())");
    assert!(c.contains("viv_arr z;"), "global decl wrong:\n{c}");
}

#[test]
fn docstrings_dropped_narrate_is_live() {
    // Docstrings still compile to nothing; `narrate "text"` is now a runtime
    // progress line (`narrate: text`) via the viv_narrate fragment.
    let c = emit_c("def f(x) {\n  \"the docstring\"\n  return x\n}\nnarrate \"hello\"\nprint(f(1.0))");
    assert!(!c.contains("docstring"), "docstring leaked into C:\n{c}");
    assert!(c.contains("viv_narrate(\"hello\")"), "narrate call missing:\n{c}");
    assert!(c.contains("static void viv_narrate"), "narrate fragment missing:\n{c}");
    assert!(c.contains("int main(void)"), "{c}");
}

#[test]
fn narrate_without_string_is_a_nop() {
    let c = emit_c("narrate quietly\nprint(1.0)");
    assert!(!c.contains("viv_narrate"), "bare narrate should compile to nothing:\n{c}");
}

#[test]
fn plot_emits_structured_line() {
    let c = emit_c("x = [1.0, 2.0, 3.0]\ny = [1.0, 4.0, 9.0]\nfigure(\"Squares\", \"n\", \"n squared\")\nplot(x, y, \"squares\")");
    assert!(c.contains("viv_plot(\"squares\", x, y)"), "plot call wrong:\n{c}");
    assert!(c.contains("static double viv_plot"), "plot fragment missing:\n{c}");
}

#[test]
fn compound_assignment_desugars() {
    let c = emit_c("x = 1.0\nx += 2.0\nx *= 3.0\nprint(x)");
    assert!(c.contains("x = (x + 2.0);"), "+= wrong:\n{c}");
    assert!(c.contains("x = (x * 3.0);"), "*= wrong:\n{c}");
}

#[test]
fn math_builtins_use_libm() {
    let c = emit_c("print(sqrt(2.0))\nprint(pow(2.0, 3.0))\nprint(exp(1.0))");
    assert!(c.contains("#include <math.h>"), "math.h missing:\n{c}");
    assert!(c.contains("sqrt(2.0)") && c.contains("pow(2.0, 3.0)") && c.contains("exp(1.0)"), "{c}");
}

/// A string is a value: it can be bound and printed.
#[test]
fn a_string_is_a_value() {
    assert!(viv::build_ir("x = \"hi\"\nprint(x)", SourceId(0)).is_ok());
    assert!(viv::interp::run("x = \"hi\"\nprint(x)\n").expect("runs").starts_with("hi\n"));
}

#[test]
fn parallel_for_emits_omp_pragma() {
    // A reduction loop and a privatized-temporary loop.
    let c = emit_c("s = 0.0\nparallel for i to 10 { s += i }\nprint(s)");
    assert!(c.contains("#pragma omp parallel for"), "no pragma:\n{c}");
    assert!(c.contains("reduction(+:s)"), "no reduction clause:\n{c}");
    assert!(c.contains("private(i)"), "loop var not privatized:\n{c}");

    let c2 = emit_c("out = zeros(4)\nparallel for i to 4 {\n  t = i * i\n  out[i] = t\n}\nprint(out[3])");
    assert!(c2.contains("private(i, t)"), "temporary not privatized:\n{c2}");
}

#[test]
fn parallel_for_runs_correctly_serial_fallback() {
    // Compiled without -fopenmp the pragma is ignored (serial), and the result
    // is still correct — the "same C runs serial or parallel" property.
    use std::process::Command;
    let c = emit_c("s = 0.0\nparallel for i to 1000 { s += i }\nprint(s)");
    let dir = std::env::temp_dir().join(format!("viv_par_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("p.c"), &c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let bin = dir.join("p");
    let compile = match Command::new(&cc)
        .args(["-std=c99", "-O2", "-Wno-unknown-pragmas"])
        .arg(dir.join("p.c"))
        .arg("-o")
        .arg(&bin)
        .output()
    {
        Ok(o) => o,
        Err(_) => return, // no C compiler
    };
    assert!(compile.status.success(), "parallel C did not compile:\n{}", String::from_utf8_lossy(&compile.stderr));
    let out = Command::new(&bin).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "499500");
}

#[test]
fn closures_lift_and_capture() {
    let c = emit_c("def boys(t_param) { return simpson(fn(t) -> exp(-t_param * t * t), 0.0, 1.0, 60) }\nprint(boys(0.5))");
    // The lambda is lifted to a top-level function taking the closure env.
    assert!(c.contains("(const double* viv_caps, double t)"), "lifted lambda signature:\n{c}");
    // The captured local is unpacked from the env.
    assert!(c.contains("double t_param = viv_caps[0];"), "capture unpack:\n{c}");
    // The call site builds a closure literal with the captured value.
    assert!(c.contains("(double[]){ t_param }"), "closure literal:\n{c}");
    assert!(c.contains("viv_simpson("), "simpson call:\n{c}");
}

#[test]
fn closure_binding_lifts_to_a_function() {
    // `f = fn(x) -> ...` then `f(2.0)`: the binding is defunctionalized into a
    // synthetic top-level def, so the compiled backends match the interpreter.
    let c = emit_c("f = fn(x) -> x * x\nprint(f(2.0))");
    assert!(c.contains("double __lam0(double x)"), "lifted lambda:\n{c}");
    assert!(c.contains("__lam0(2.0)"), "call through the binding:\n{c}");
}

#[test]
fn def_passed_as_value_specializes() {
    // A named def passed to a function specializes the callee (Theorem: after
    // defunctionalization the backends only see first-order calls).
    let c = emit_c(
        "def triple(x) { return x * 3.0 }\n\
         def apply_twice(f, x) { return f(f(x)) }\n\
         print(apply_twice(triple, 2.0))",
    );
    assert!(c.contains("double apply_twice__triple(double x)"), "specialization:\n{c}");
    assert!(c.contains("triple(triple(x))"), "substituted body:\n{c}");
    assert!(c.contains("apply_twice__triple(2.0)"), "rewritten call:\n{c}");
}

#[test]
fn function_arg_forwarding_specializes_transitively() {
    // A param that is only forwarded (never called) still counts as
    // function-typed; the chain outer -> apply_twice specializes end to end.
    let c = emit_c(
        "def triple(x) { return x * 3.0 }\n\
         def apply_twice(f, x) { return f(f(x)) }\n\
         def outer(f, x) { return apply_twice(f, x) + 1.0 }\n\
         print(outer(triple, 2.0))",
    );
    assert!(c.contains("double outer__triple(double x)"), "outer specialization:\n{c}");
    assert!(c.contains("apply_twice__triple("), "inner specialization:\n{c}");
}

#[test]
fn capturing_lambda_is_lifted_with_its_captures_as_parameters() {
    // A lambda that reads a local of its def captures it by value:
    // the lifted def takes it as a trailing
    // parameter, the specialization of `apply` passes it on, and no closure
    // environment is allocated.
    let c = emit_c(
        "def apply(f, x) { return f(x) }\n\
         def demo(a) { g = fn(u) -> u + a\n  return apply(g, 1.0) }\n\
         print(demo(3.0))",
    );
    assert!(c.contains("double __lam0(double u, double a)"), "lifted lambda:\n{c}");
    assert!(c.contains("apply____lam0(1.0, a)") || c.contains("apply__lam0(1.0, a)"), "capture passed at the call:\n{c}");
}

#[test]
fn a_closure_cannot_be_returned_or_stored() {
    for src in [
        "def mk(a) {\n  g = fn(x) -> a * x\n  return g\n}\nshow mk(2.0)",
        "def mk(a) {\n  g = fn(x) -> a * x\n  gs = [g, g]\n  return 1.0\n}\nshow mk(2.0)",
    ] {
        let err = viv::build_ir(src, SourceId(0)).unwrap_err();
        assert!(err.iter().any(|d| d.message.contains("`g` is a closure")), "{err:?}");
    }
    let err = viv::build_ir("def mk(a) {\n  return fn(x) -> a * x\n}\nshow mk(2.0)", SourceId(0)).unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("it cannot be returned or stored")), "{err:?}");
}

#[test]
fn infers_array_param_and_return_types() {
    // No annotations: `a`/`b` are arrays (indexed), `scale_vec` returns an array
    // (it returns `zeros`), and `k` is a number (passed 10.0).
    let c = emit_c(
        "def dot(a, b) {\n  s = 0.0\n  for i to 3 { s = s + a[i] * b[i] }\n  return s\n}\n\
         def scale_vec(v, k) {\n  r = zeros(3)\n  for i to 3 { r[i] = v[i] * k }\n  return r\n}\n\
         u = [1.0, 2.0, 3.0]\nprint(dot(u, u))\ny = scale_vec(u, 2.0)\nprint(y[0])",
    );
    assert!(c.contains("double dot(viv_arr a_viv_in, viv_arr b_viv_in)"), "dot signature:\n{c}");
    assert!(c.contains("viv_arr scale_vec(viv_arr v_viv_in, double k)"), "scale_vec signature:\n{c}");
}

#[test]
fn infers_array_param_from_call_site_backward() {
    // `x` is constrained only by being passed to `dot` (which needs arrays).
    let c = emit_c(
        "def dot(a, b) {\n  s = 0.0\n  for i to 2 { s = s + a[i] * b[i] }\n  return s\n}\n\
         def nsq(x) { return dot(x, x) }\nu = [3.0, 4.0]\nprint(nsq(u))",
    );
    assert!(c.contains("double nsq(viv_arr x_viv_in)"), "x should be inferred array:\n{c}");
}

#[test]
fn matmul_and_transpose_lower_to_runtime() {
    let c = emit_c("A = [[1.0, 2.0], [3.0, 4.0]]\nB = transpose(A) @ A\nprint(B[0, 0])");
    assert!(c.contains("viv_matmul("), "matmul not lowered:\n{c}");
    assert!(c.contains("viv_transpose("), "transpose not lowered:\n{c}");
    // `*` here was never used, so elementwise/scale runtime must be pruned.
    assert!(!c.contains("viv_arr_scale"), "scale runtime should be pruned:\n{c}");
}

#[test]
fn scalar_times_array_is_scale_not_matmul() {
    let c = emit_c("A = [[1.0, 2.0], [3.0, 4.0]]\nB = 2.0 * A\nprint(B[0, 0])");
    assert!(c.contains("viv_arr_scale("), "scalar*array should be a scale:\n{c}");
    assert!(!c.contains("viv_matmul"), "{c}");
}

#[test]
fn scalar_broadcast_over_array_compiles() {
    // `A + 1.0` broadcasts the scalar elementwise (interp and C agree: 2).
    // This test previously asserted a rejection, from before broadcast landed.
    let c = emit_c("A = [[1.0]]\nx = A + 1.0\nprint(x[0, 0])");
    assert!(c.contains("viv_arr"), "array code:\n{c}");
}

#[test]
fn rejects_indexing_a_number() {
    let err = viv::build_ir("x = 1.0\nprint(x[0])", SourceId(0)).unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("cannot index a non-array")));
}

#[test]
fn rejects_non_boolean_condition() {
    let err = viv::build_ir("def f(n) { if n { return 1.0 } return 0.0 }", SourceId(0)).unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("condition must be a boolean")));
}

#[test]
fn rejects_break_outside_loop() {
    let err = viv::build_ir("def f(n) { break }", SourceId(0)).unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("`break` outside")));
}

#[test]
fn structurally_identical_subexprs_share_identity() {
    // `r * r` builds two `Var{r}` nodes that are structurally identical, so the
    // content-addressed store (1.3) holds one. Distinct nodes:
    // Func, Return, Bin(outer), Bin(inner), Num(3.0), Var(r) => 6.
    let program = viv::build_ir("def a(r) { return 3.0 * r * r }", SourceId(0)).unwrap();
    assert_eq!(program.len(), 6, "expected the two `r` refs to share one NodeId");
}

#[test]
fn extended_libm_maps_to_c_names() {
    let c = emit_c("print(atan2(1.0, 2.0))\nprint(gamma(5.0))\nprint(mod(7.0, 3.0))\nprint(erf(1.0))\nprint(sinh(1.0))");
    assert!(c.contains("atan2(1.0, 2.0)"), "{c}");
    assert!(c.contains("tgamma(5.0)"), "gamma must lower to tgamma:\n{c}");
    assert!(c.contains("fmod(7.0, 3.0)"), "mod must lower to fmod:\n{c}");
    assert!(c.contains("erf(1.0)") && c.contains("sinh(1.0)"), "{c}");
    assert!(c.contains("#include <math.h>"), "{c}");
}

#[test]
fn spectrum_sym_emits_sweep_and_release_paths() {
    // The contracted solve compiles BOTH the debug sweep (Jacobi + QL,
    // cross-checked, loud abort on dissent) and the release fast path,
    // switched by -DVIV_RELEASE at C build time.
    let c = emit_c("A = zeros(2, 2)\nA[0,0] = 2.0\nA[1,1] = 3.0\nev = zeros(2)\nspectrum_sym(A, ev)\nshow ev");
    assert!(c.contains("#ifndef VIV_RELEASE"), "mode switch:\n{c}");
    assert!(c.contains("viv_tridiag_eigvals"), "QL method:\n{c}");
    assert!(c.contains("viv_eig_sym"), "jacobi method:\n{c}");
    assert!(c.contains("SOLVER SWEEP DISSENT"), "dissent report:\n{c}");
}

#[test]
fn spectrum_sym_interp_debug_and_release_agree() {
    let src = "A = zeros(2, 2)\nA[0,0] = 2.0\nA[0,1] = 1.0\nA[1,0] = 1.0\nA[1,1] = 3.0\nev = zeros(2)\nspectrum_sym(A, ev)\nshow ev[0]\nshow ev[1]";
    let dbg = viv::interp::run(src).expect("debug run");
    let rel = viv::interp::run_release(src).expect("release run");
    // exact spectrum: (5 +/- sqrt(5))/2
    assert!(dbg.contains("1.38197"), "debug values: {dbg}");
    assert_eq!(
        dbg.lines().take(2).collect::<Vec<_>>(),
        rel.lines().take(2).collect::<Vec<_>>(),
        "debug and release must agree on a well-posed problem"
    );
}

#[test]
fn spectrum_sym_interp_rejects_asymmetric_in_debug() {
    let src = "A = zeros(2, 2)\nA[0,1] = 5.0\nA[1,0] = 2.0\nev = zeros(2)\nspectrum_sym(A, ev)";
    let err = viv::interp::run(src).unwrap_err();
    assert!(err.contains("contract violation"), "{err}");
    // release skips the check by design
    assert!(viv::interp::run_release(src).is_ok());
}

#[test]
fn dense_la_builtins_emit_runtime() {
    let c = emit_c(
        "A = zeros(2, 2)\nA[0,0] = 2.0\nA[1,1] = 3.0\nb = zeros(2)\nb[0] = 1.0\nb[1] = 1.0\n\
         x = zeros(2)\nlin_solve(A, b, x)\nL = zeros(2, 2)\ncholesky(A, L)\n\
         wr = zeros(2)\nwi = zeros(2)\neig_general(A, wr, wi)\nshow x",
    );
    assert!(c.contains("viv_lin_solve"), "lin_solve runtime:\n{c}");
    assert!(c.contains("viv_cholesky"), "cholesky runtime:\n{c}");
    assert!(c.contains("viv_eig_general"), "eig_general runtime:\n{c}");
    assert!(c.contains("SOLVER CERTIFICATE FAILED"), "debug certificates present:\n{c}");
}

#[test]
fn eig_general_interp_complex_pair() {
    // 2x2 rotation-like block: eigenvalues 1 +/- 2i.
    let src = "A = zeros(2, 2)\nA[0,0] = 1.0\nA[0,1] = -2.0\nA[1,0] = 2.0\nA[1,1] = 1.0\n\
               wr = zeros(2)\nwi = zeros(2)\neig_general(A, wr, wi)\n\
               show wr[0]\nshow wi[0]\nshow wi[1]";
    let out = viv::interp::run(src).expect("run");
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines[0], "1", "re: {out}");
    assert_eq!(lines[1], "-2", "im low: {out}");
    assert_eq!(lines[2], "2", "im high: {out}");
}

#[test]
fn cholesky_interp_rejects_indefinite() {
    let src = "A = zeros(2, 2)\nA[0,0] = 1.0\nA[1,1] = -1.0\nL = zeros(2, 2)\ncholesky(A, L)";
    let err = viv::interp::run(src).unwrap_err();
    assert!(err.contains("not positive definite"), "{err}");
}

#[test]
fn lin_solve_multi_with_row_pivoting() {
    // Regression: the packed-LU solve must apply ALL pivots before forward
    // substitution. The original interleaved order was wrong the moment a row
    // swap fired — caught by the residual certificate on this exact matrix.
    let src = "A = zeros(3, 3)\nA[0,0] = 1.0\nA[0,1] = 4.0\nA[0,2] = 2.0\n\
               A[1,0] = 3.0\nA[1,1] = 1.0\nA[1,2] = 1.0\n\
               A[2,0] = 2.0\nA[2,1] = 5.0\nA[2,2] = 9.0\n\
               B = zeros(3, 1)\nB[0,0] = 1.0\nB[1,0] = 2.0\nB[2,0] = 3.0\n\
               X = zeros(3, 1)\nlin_solve_multi(A, B, X)\nshow X[0,0]\nshow X[2,0]";
    let out = viv::interp::run(src).expect("pivoting solve must pass its certificate");
    assert!(out.starts_with("0.6\n0.2"), "numpy: x = [0.6, 0, 0.2]; got {out}");
}

#[test]
fn eig_near_finds_nearest_of_a_normal_matrix() {
    let src = "G = zeros(4, 4)\nG[0,0] = 1.0\nG[0,1] = 2.0\nG[0,3] = 1.0\n\
               G[1,0] = -3.0\nG[1,1] = 1.0\nG[1,2] = 4.0\nG[2,1] = -2.0\n\
               G[2,2] = 1.0\nG[2,3] = 5.0\nG[3,0] = 1.0\nG[3,2] = -1.0\nG[3,3] = 2.0\n\
               re = zeros(2)\nim = zeros(2)\neig_near(G, 3.0, 2, re, im)\nshow re[0]\nshow re[1]";
    let out = viv::interp::run(src).expect("run");
    assert!(out.starts_with("2.27553\n0.490657"), "{out}");
}

#[test]
fn checkpoint_saves_and_resumes_identically() {
    // checkpoint() markers: a full run with VIV_CKPT_DIR snapshots all
    // top-level state at each marker; VIV_RESUME=k + VIV_CKPT_IN skips the
    // statements before marker k, loads the snapshot, and the suffix output
    // is byte-identical to the full run's. The per-cell notebook cache
    // depends on exactly this property.
    use std::process::Command;
    let src = "a = 2.0\nA = zeros(3)\nfor i to 3 { A[i] = a * (i + 1.0) }\nshow a\ncheckpoint()\nb = a * 10.0\nshow b\nshow A[2]\ncheckpoint()\nc = b + A[1]\nshow c\n";
    let c = emit_c(src);
    let dir = std::env::temp_dir().join(format!("viv_ckpt_{}", std::process::id()));
    let ck = dir.join("ck");
    std::fs::create_dir_all(&ck).unwrap();
    std::fs::write(dir.join("p.c"), &c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let bin = dir.join("p");
    let compile = match Command::new(&cc)
        .args(["-std=c99", "-O2", "-Wno-unknown-pragmas"])
        .arg(dir.join("p.c"))
        .arg("-o")
        .arg(&bin)
        .output()
    {
        Ok(o) => o,
        Err(_) => return, // no C compiler
    };
    assert!(compile.status.success(), "ckpt C did not compile:\n{}", String::from_utf8_lossy(&compile.stderr));
    let full = Command::new(&bin).env("VIV_CKPT_DIR", &ck).output().unwrap();
    let full_out = String::from_utf8_lossy(&full.stdout).to_string();
    assert_eq!(full_out, "a: 2\nb: 20\n6\nc: 24\n", "full run output:\n{full_out}");
    let r0 = Command::new(&bin)
        .env("VIV_RESUME", "0")
        .env("VIV_CKPT_IN", ck.join("ck_0.bin"))
        .output()
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&r0.stdout), "b: 20\n6\nc: 24\n");
    let r1 = Command::new(&bin)
        .env("VIV_RESUME", "1")
        .env("VIV_CKPT_IN", ck.join("ck_1.bin"))
        .output()
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&r1.stdout), "c: 24\n");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn session_run_cell_matches_full_runs() {
    // run_cell with a matching prefix executes ONLY the cell on retained
    // state, and its output equals the cell's share of a full prefix+cell run.
    use viv::interp::{run, run_cell};
    let c0 = "a = 2.0\nA = zeros(3)\nfor i to 3 { A[i] = a * (i + 1.0) }\nshow a";
    let c1 = "b = a * 10.0\nshow b\nshow A[2]";
    let c2 = "c = b + A[1]\nshow c";
    // cold: cell 0 with empty prefix
    assert_eq!(run_cell("", c0).unwrap(), "a: 2\n");
    // continues: cells 1 and 2 run incrementally
    assert_eq!(run_cell(c0, c1).unwrap(), "b: 20\n6\n");
    let p2 = format!("{c0}\n{c1}");
    assert_eq!(run_cell(&p2, c2).unwrap(), "c: 24\n");
    // a mismatched prefix (edited cell) cold-rebuilds and still agrees with a
    // fresh full run of the same program
    let c1b = "b = a * 100.0\nshow b";
    assert_eq!(run_cell(c0, c1b).unwrap(), "b: 200\n");
    let full = run(&format!("{c0}\n{c1b}\n{c2}")).unwrap();
    assert!(full.ends_with("c: 204\n"), "full run tail: {full}");
    let p2b = format!("{c0}\n{c1b}");
    assert_eq!(run_cell(&p2b, c2).unwrap(), "c: 204\n");
}

#[test]
fn run_cell_carries_the_random_state_from_cell_to_cell() {
    // rand() continues where the previous cell left it, whether the session
    // continues or is rebuilt from the prefix: every cell draws what the whole
    // program draws at that point
    use viv::interp::{run, run_cell};
    let c0 = "srand(7.0)\nu = rand()\nshow u";
    let c1 = "v = rand()\nshow v";
    let full = run(&format!("{c0}\n{c1}")).unwrap();
    let a = run_cell("", c0).unwrap();
    let b = run_cell(c0, c1).unwrap();
    assert_eq!(format!("{a}{b}"), full);
    // a session for another program first, so the next call rebuilds from the prefix
    run_cell("", "w = 1.0").unwrap();
    assert_eq!(run_cell(c0, c1).unwrap(), b);
    // with no srand at all, the second cell continues the default seed
    let d0 = "p = rand()\nshow p";
    let full = run(&format!("{d0}\n{c1}")).unwrap();
    let a = run_cell("", d0).unwrap();
    assert_eq!(format!("{a}{}", run_cell(d0, c1).unwrap()), full);
}

#[test]
fn scene_wire_verts_emit_protocol_lines() {
    // The scene protocol: scene()/wire()/verts() lower to
    // runtime calls printing the same lines the interpreter emits.
    let src = "n = 3\nx = zeros(n)\ny = zeros(n)\nz = zeros(n)\nv = zeros(n)\nfor i to n {\n    x[i] = i * 1.0\n    y[i] = i * 2.0\n    z[i] = i * 3.0\n    v[i] = 9.0 - i\n}\nscene(\"demo\")\nwire(x, y, z, \"frame\")\nwire(x, y, z, v, \"stressed\")\nverts(x, y, z, \"joints\")";
    let c = emit_c(src);
    assert!(c.contains("static double viv_scene("), "scene runtime missing:\n{c}");
    assert!(c.contains("viv_scene(\"demo\")"), "scene call missing:\n{c}");
    assert!(c.contains("viv_sc3(\"wire\", \"frame\""), "3-array wire missing:\n{c}");
    assert!(c.contains("viv_sc4(\"wire\", \"stressed\""), "4-array wire missing:\n{c}");
    assert!(c.contains("viv_sc3(\"verts\", \"joints\""), "verts missing:\n{c}");

    // And the interpreter emits the identical protocol.
    let out = viv::interp::run(src).expect("interp run");
    assert!(out.contains("scene: demo"), "{out}");
    assert!(out.contains("wire: frame ; [0, 1, 2] ; [0, 2, 4] ; [0, 3, 6]"), "{out}");
    assert!(out.contains("wire: stressed ; [0, 1, 2] ; [0, 2, 4] ; [0, 3, 6] ; [9, 8, 7]"), "{out}");
    assert!(out.contains("verts: joints ; [0, 1, 2] ; [0, 2, 4] ; [0, 3, 6]"), "{out}");
}

/// A user variable named `fmax` is legal Viv — the language spells that builtin
/// `max` — but `fmax` is a libm function, so emitting it verbatim produced
/// `double fmax;` on top of math.h's declaration and the C would not compile.
/// Found by compiling every example deck (hasegawa_wakatani.viv).
#[test]
fn user_names_that_collide_with_libm_are_escaped() {
    let src = r#"
fmax = 0.0
fmin = 1.0
hypot = 2.0
for i to 3 {
    if i > fmax {
        fmax = i
    }
}
show fmax
show fmin
show hypot
"#;
    let c = viv::compile_c(src).expect("collliding names should compile");
    // the declarations must be escaped...
    assert!(c.contains("double fmax_;"), "fmax not escaped:\n{c}");
    assert!(c.contains("double fmin_;"), "fmin not escaped:\n{c}");
    assert!(c.contains("double hypot_;"), "hypot not escaped:\n{c}");
    // ...and so must every use, or the C would not even parse
    assert!(!c.contains("double fmax;"), "unescaped declaration remains:\n{c}");
    assert!(c.contains("fmax_ = i"), "use site not escaped:\n{c}");
    // the label the program shows is the name the USER wrote, not the C one
    assert!(c.contains("\"fmax\""), "show label should keep the Viv name:\n{c}");
}

/// A `def` named with the C spelling of a Viv builtin used to make the compiled
/// program call the wrong function, silently: with `def fmod` present,
/// `mod(7, 3)` returned 10 compiled and 1 interpreted. Refused now, because a
/// silent divergence between the engines is the one outcome the three-backend
/// discipline exists to prevent.
#[test]
fn a_def_named_for_an_emitted_c_builtin_is_refused() {
    for (name, viv) in [("fmod", "mod"), ("fmin", "min"), ("fmax", "max"), ("tgamma", "gamma")] {
        let src = format!("def {name}(a, b) {{\n    return a + b\n}}\nshow {name}(1.0, 2.0)\n");
        let err = viv::compile_c(&src).expect_err(&format!("`def {name}` must be refused"));
        assert!(err.contains(name) && err.contains(viv), "unhelpful message for {name}: {err}");
    }
}

/// ...while a VARIABLE of the same name is ordinary Viv and must keep working,
/// with the two engines agreeing. `fmax` is the real case: Viv spells that
/// builtin `max`, so `fmax = 0.0` appears in real programs
/// (hasegawa_wakatani.viv) and has to survive the trip through C.
#[test]
fn variables_named_for_libm_functions_agree_across_engines() {
    let src = "fmax = 7.0\nfmod = 2.0\nhypot = 3.0\nshow max(2.0, fmax)\nshow mod(7.0, 3.0)\nshow fmax + fmod + hypot\n";
    let interp = viv::interp::run(src).expect("should interpret");
    let c = viv::compile_c(src).expect("should compile");
    // the emitted C escapes the variables but not the builtin calls
    assert!(c.contains("double fmax_;"), "variable not escaped:\n{c}");
    assert!(c.contains("fmax(2.0, fmax_)"), "builtin call was escaped too:\n{c}");
    assert!(c.contains("fmod(7.0, 3.0)"), "builtin call was escaped too:\n{c}");
    // and the interpreter's answers are the ones the C computes
    let nums: Vec<&str> = interp.lines().filter(|l| !l.starts_with("__")).collect();
    assert_eq!(nums, vec!["7", "1", "12"], "interpreter changed: {interp}");
}

/// A `test "name" { .. }` block reports under its NAME. Before this, every
/// assertion printed an anonymous `assert_near PASS` and the name the author
/// wrote was discarded by the parser, so a run of twenty tests gave twenty
/// identical lines and told you nothing about which claim held.
#[test]
fn a_test_block_reports_under_its_name() {
    let src = r#"
test "simpson integrates exp correctly" {
    assert_near(simpson(fn(x) -> exp(x), 0.0, 1.0, 60), exp(1.0) - 1.0, 1.0e-9)
}
"#;
    let interp = viv::interp::run(src).expect("should interpret");
    assert!(
        interp.contains("PASS simpson integrates exp correctly"),
        "interpreter did not name the test: {interp}"
    );
    assert!(!interp.contains("assert_near PASS"), "per-assert line leaked: {interp}");
    // the C backend emits the same reporting calls
    let c = viv::compile_c(src).expect("should compile");
    assert!(c.contains("viv_test_begin(\"simpson integrates exp correctly\")"), "{c}");
    assert!(c.contains("viv_test_end()"), "{c}");
}

/// Several asserts in one block collapse to one line, because a test is the
/// unit that passes or fails; a failure names the test and carries the first
/// failing assert's detail. An assert OUTSIDE any test keeps its own line,
/// since there is no name to report under.
#[test]
fn test_blocks_collapse_and_failures_name_themselves() {
    let src = r#"
test "two asserts, one verdict" {
    assert_near(1.0, 1.0, 1.0e-9)
    assert_gt(2.0, 1.0)
}
test "a failing one" {
    assert_near(1.0, 2.0, 1.0e-9)
}
test "nothing asserted" {
}
assert_near(3.0, 3.0, 1.0e-9)
"#;
    let (out, fails) = viv::interp::run_checked(src, false).expect("should interpret");
    let lines: Vec<&str> = out.lines().filter(|l| !l.starts_with("__")).collect();
    assert_eq!(
        lines,
        vec![
            "PASS two asserts, one verdict",
            "FAIL a failing one -- assert_near: |1 - 2| = 1 > 1e-09",
            "PASS nothing asserted (no assertions)",
            "assert_near PASS",
        ],
        "unexpected report:\n{out}"
    );
    assert_eq!(fails, 1, "the failing assert must still count toward the exit code");
}

/// `show` on a rank-2 array lines its rows up under the first one. The indent
/// is the label's width, so the matrix reads as a matrix instead of having
/// every row after the first sit flush against the left margin.
#[test]
fn show_aligns_a_matrix_under_its_label() {
    let src = "hamiltonian = [[1.0, 2.0], [3.0, 4.0]]\nshow hamiltonian\nshow [[5.0, 6.0], [7.0, 8.0]]\n";
    let out = viv::interp::run(src).expect("should interpret");
    let want = "hamiltonian: [[1, 2],\n              [3, 4]]\n[[5, 6],\n [7, 8]]\n";
    assert_eq!(out, want, "matrix rows are not aligned:\n{out}");
    // "hamiltonian: [" is 14 chars, so row 2 pads to column 14
    assert!(out.contains("\n              [3, 4]]"), "{out}");
}

/// Every IR kind must have its own tag. The tag is the first thing hashed into
/// a NodeId, so two kinds sharing one means their content addresses are
/// distinguished only by their payload bytes happening to differ — which is
/// not a property anything checks. `Scatter` and `MapMath` both claimed 36
/// until this test existed.
///
/// Reading the source beats enumerating the variants: a new kind is caught
/// automatically, without anyone remembering to add it here.
#[test]
fn every_ir_kind_has_a_distinct_tag() {
    let src = include_str!("../src/lower.rs");
    let start = src.find("fn tag(&self)").expect("tag() should exist");
    let body = &src[start..];
    let end = body.find("\n    }\n").expect("tag() should end");
    let body = &body[..end];

    let mut by_tag: std::collections::BTreeMap<u32, Vec<&str>> = Default::default();
    for line in body.lines() {
        let Some((lhs, rhs)) = line.split_once("=>") else { continue };
        let Some(kind) = lhs.trim().strip_prefix("Viv::") else { continue };
        let kind = kind.split(|c: char| !c.is_alphanumeric()).next().unwrap_or("");
        let Ok(tag) = rhs.trim().trim_end_matches(',').parse::<u32>() else { continue };
        by_tag.entry(tag).or_default().push(kind);
    }
    assert!(by_tag.len() > 40, "only parsed {} tags — did tag() change shape?", by_tag.len());
    let dups: Vec<String> = by_tag
        .iter()
        .filter(|(_, kinds)| kinds.len() > 1)
        .map(|(t, kinds)| format!("tag {t}: {}", kinds.join(", ")))
        .collect();
    assert!(dups.is_empty(), "IR kinds sharing a tag:\n  {}", dups.join("\n  "));
}

/// Content addresses are a PROMISE: `viv diff` calls two definitions the same
/// when their bodies hash the same, and the ladder tutorials on the site key
/// their rung cards off those hashes. So a refactor that silently moves them
/// breaks published artifacts, quietly.
///
/// These are golden values for a program that exercises arrays, a def with
/// parameters, and top-level globals. They were minted before `Ty::Arr` gained
/// an element type and must survive it: `Arr(Num)` contributes no element byte
/// to the hash, exactly so that every address minted before elements existed
/// still holds. A new element type SHOULD change `Arr(Cplx)` addresses -- and
/// must leave these alone.
#[test]
fn array_content_addresses_are_stable() {
    let src = "def scale(v, s) {\n    n = len(v)\n    out = zeros(n)\n    for i to n {\n        out[i] = v[i] * s\n    }\n    return out\n}\nx = [1.0, 2.0, 3.0]\ny = scale(x, 2.0)\nshow y\n";
    let report = viv::hash_report(src).expect("should hash");
    for want in [
        "e51fbb35  body af1349b9  global x",
        "3c04cc5f  body af1349b9  global y",
        "cb30a301  body 666a9859  def scale(2 params)",
        "df1cd85c  body c02879a4  top-level program",
    ] {
        assert!(
            report.contains(want),
            "a content address moved.\n  want: {want}\n  got:\n{report}"
        );
    }
}

// ---- complex arrays and the FFT --------------------------------------------

/// Compile `src` to C, build it, run it, return stdout. `None` if there is no
/// C compiler, so the suite still passes on a machine without one.
fn run_compiled(src: &str, tag: &str) -> Option<String> {
    let c = viv::compile_c(src).expect("should compile");
    let dir = std::env::temp_dir().join(format!("viv_e2e_{}_{tag}", std::process::id()));
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

/// The two engines must print the SAME spectrum, to the byte, at a power of
/// two (radix-2), a prime (Bluestein) and a composite length. This is the
/// parity claim for the transform: the C runtime's viv_fft is the
/// interpreter's fft_in_place line for line, over the same interleaved memory.
#[test]
fn fft_agrees_across_engines_at_every_kind_of_length() {
    let src = r#"
def spectrum(n) {
    x = czeros(n)
    for j to n {
        x[j] = complex(sin(0.7 * j) + 0.3 * j, cos(1.3 * j))
    }
    fft(x)
    return x
}
a = spectrum(8)
show a
b = spectrum(7)
show b
c = spectrum(12)
show c
"#;
    let interp = interp_lines(src);
    assert!(interp.starts_with("a: [9.02305+0.236016i"), "unexpected spectrum:\n{interp}");
    if let Some(compiled) = run_compiled(src, "fftparity") {
        assert_eq!(interp, compiled, "the engines disagree on a spectrum");
    }
}

/// Correctness, not just agreement: against the O(N^2) definition, the round
/// trip, and Parseval -- at lengths that exercise both algorithms.
#[test]
fn fft_matches_its_definition_and_inverts() {
    let src = r#"
PI = 3.141592653589793
def err_vs_direct(n) {
    x = czeros(n)
    for j to n { x[j] = complex(sin(0.7 * j) + 0.3 * j, cos(1.3 * j)) }
    ref = czeros(n)
    for k to n {
        sr = 0.0
        si = 0.0
        for j to n {
            ang = 0.0 - 2.0 * PI * k * j / n
            sr = sr + re(x[j]) * cos(ang) - im(x[j]) * sin(ang)
            si = si + re(x[j]) * sin(ang) + im(x[j]) * cos(ang)
        }
        ref[k] = complex(sr, si)
    }
    fft(x)
    worst = 0.0
    big = 0.0
    for k to n {
        if abs(x[k] - ref[k]) > worst { worst = abs(x[k] - ref[k]) }
        if abs(ref[k]) > big { big = abs(ref[k]) }
    }
    return worst / big
}
def roundtrip(n) {
    x = czeros(n)
    y = czeros(n)
    for j to n {
        x[j] = complex(sin(0.7 * j), cos(1.3 * j))
        y[j] = x[j]
    }
    fft(y)
    ifft(y)
    worst = 0.0
    for j to n { if abs(y[j] - x[j]) > worst { worst = abs(y[j] - x[j]) } }
    return worst
}
test "radix-2 matches the definition" { assert_lt(err_vs_direct(16), 1.0e-13) }
test "Bluestein matches it at a prime" { assert_lt(err_vs_direct(13), 1.0e-13) }
test "Bluestein matches it at a composite" { assert_lt(err_vs_direct(30), 1.0e-13) }
test "ifft inverts fft, power of two" { assert_lt(roundtrip(64), 1.0e-13) }
test "ifft inverts fft, prime" { assert_lt(roundtrip(97), 1.0e-12) }
"#;
    let (out, fails) = viv::interp::run_checked(src, false).expect("should interpret");
    assert_eq!(fails, 0, "an FFT property failed:\n{out}");
    assert_eq!(out.matches("PASS ").count(), 5, "expected five passing tests:\n{out}");
}

/// A complex array passed to a function that INDEXES its parameter, with the
/// callee defined first. Indexing alone says "array" but not of what, and a
/// naive `Arr(Num)` guess would lock in the wrong element type before the call
/// site got to say otherwise. The inference holds that guess back for a first
/// fixpoint so real evidence wins, in either definition order.
#[test]
fn a_complex_array_parameter_is_inferred_in_either_order() {
    let src = r#"
def power(spec) {
    tot = 0.0
    for k to len(spec) { tot = tot + abs(spec[k]) * abs(spec[k]) }
    return tot
}
def via(spec) { return power(spec) }
x = czeros(4)
for j to 4 { x[j] = complex(j, 0.0 - j) }
show power(x)
show via(x)
def total(v) {
    s = 0.0
    for i to len(v) { s = s + v[i] }
    return s
}
show total([1.0, 2.0, 3.0])
"#;
    let interp = interp_lines(src);
    assert_eq!(interp, "28\n28\n6\n");
    if let Some(compiled) = run_compiled(src, "cparam") {
        assert_eq!(interp, compiled);
    }
}

/// Complex and real arrays are different types, and the refusals say so. A
/// complex array is its own interpreter value rather than a flag on the real
/// one, precisely so a real-array builtin CANNOT read interleaved (re, im)
/// data as if it were reals -- it fails instead of returning nonsense.
#[test]
fn complex_and_real_arrays_do_not_mix_silently() {
    for (src, want) in [
        ("x = zeros(8)\nfft(x)\n", "needs a complex array"),
        ("x = zeros(4)\nx[0] = 1.0 + 2.0i\n", "cannot store a complex number in a real array"),
        ("x = czeros(4)\ny = sin(x)\n", "complex array"),
    ] {
        let e = viv::interp::run(src).expect_err("interpreter should refuse");
        assert!(e.contains(want), "interpreter said: {e}");
        assert!(viv::compile_c(src).is_err(), "the C backend accepted:\n{src}");
    }
}

/// An array's ELEMENT type is part of its content address. If it were not,
/// `zeros(4)` and `czeros(4)` stored in the same place would hash alike and
/// `viv diff` would call two different programs the same.
#[test]
fn a_complex_array_does_not_share_a_content_address_with_a_real_one() {
    let real = viv::hash_report("def f(v: Arr) {\n    return len(v)\n}\n").unwrap();
    let cplx = viv::hash_report("def f(v: CArr) {\n    return len(v)\n}\n").unwrap();
    let addr = |r: &str| r.lines().find(|l| l.contains("def f")).unwrap().to_string();
    assert_ne!(addr(&real), addr(&cplx), "element type is not in the hash");
}

// ---- the builtins a plane-wave DFT code needed -----------------------------

/// eig_herm by its certificates, in BOTH engines: the residual |A V - V w|,
/// orthonormality |V^H V - I|, the trace, and a DEGENERATE spectrum. The
/// degenerate case is the one that matters: embedding a Hermitian matrix in a
/// real symmetric one of twice the size returns linearly dependent complex
/// vectors inside a repeated eigenvalue, which is why this is a native complex
/// Householder + QL instead.
#[test]
fn eig_herm_passes_its_certificates_in_both_engines() {
    let src = r#"
def residual(A, w, V, n) {
    worst = 0.0
    for c to n {
        for i to n {
            s = complex(0.0, 0.0)
            for j to n { s = s + A[i, j] * V[j, c] }
            if abs(s - w[c] * V[i, c]) > worst { worst = abs(s - w[c] * V[i, c]) }
        }
    }
    return worst
}
def ortho_gap(V, n) {
    worst = 0.0
    for a to n {
        for b to n {
            s = complex(0.0, 0.0)
            for i to n { s = s + conj(V[i, a]) * V[i, b] }
            want = 0.0
            if a == b { want = 1.0 }
            if abs(s - want) > worst { worst = abs(s - want) }
        }
    }
    return worst
}
n = 16
A = czeros(n, n)
srand(11.0)
for i to n {
    A[i, i] = 2.0 * rand() - 1.0
    for j to i {
        z = complex(2.0 * rand() - 1.0, 2.0 * rand() - 1.0)
        A[i, j] = z
        A[j, i] = conj(z)
    }
}
w = zeros(n)
V = czeros(n, n)
eig_herm(A, w, V)
test "dense Hermitian: residual and orthonormality" {
    assert_lt(residual(A, w, V, n), 1.0e-12)
    assert_lt(ortho_gap(V, n), 1.0e-13)
}
"#;
    // The degenerate case: rotate diag(1, 1, 1, 4, 4, 9) by a genuinely unitary
    // matrix (the eigenvectors of another Hermitian matrix), then diagonalize.
    let src2 = r#"
m = 6
H6 = czeros(m, m)
for i to m {
    H6[i, i] = 0.3 * i
    for j to i {
        z = complex(sin(1.7 * i + 0.3 * j), cos(0.9 * i - 1.1 * j))
        H6[i, j] = z
        H6[j, i] = conj(z)
    }
}
uw = zeros(m)
U = czeros(m, m)
eig_herm(H6, uw, U)
dvals = [1.0, 1.0, 1.0, 4.0, 4.0, 9.0]
D = czeros(m, m)
for i to m {
    for j to m {
        s = complex(0.0, 0.0)
        for k to m { s = s + U[i, k] * dvals[k] * conj(U[j, k]) }
        D[i, j] = s
    }
}
dw = zeros(m)
dvec = czeros(m, m)
eig_herm(D, dw, dvec)
show dw
worst = 0.0
for a to m {
    for b to m {
        s = complex(0.0, 0.0)
        for i to m { s = s + conj(dvec[i, a]) * dvec[i, b] }
        want = 0.0
        if a == b { want = 1.0 }
        if abs(s - want) > worst { worst = abs(s - want) }
    }
}
test "a repeated eigenvalue still gets orthonormal eigenvectors" {
    assert_near(dw[0], 1.0, 1.0e-12)
    assert_near(dw[2], 1.0, 1.0e-12)
    assert_near(dw[4], 4.0, 1.0e-12)
    assert_near(dw[5], 9.0, 1.0e-12)
    assert_lt(worst, 1.0e-13)
}
"#;
    for (tag, program) in [("herm_dense", src), ("herm_degen", src2)] {
        let interp = interp_lines(program);
        assert!(interp.contains("PASS ") && !interp.contains("FAIL "), "interpreter:\n{interp}");
        if let Some(compiled) = run_compiled(program, tag) {
            assert!(compiled.contains("PASS ") && !compiled.contains("FAIL "), "compiled:\n{compiled}");
        }
    }
}

/// fft3 is three passes of fft, so it inherits Bluestein: a 6 x 5 x 8 grid is
/// not a power of two on two of its axes. A single plane wave must land in
/// exactly one bin with height N0 N1 N2, and ifft3 must undo it.
#[test]
fn fft3_transforms_a_plane_wave_to_one_bin_on_an_awkward_grid() {
    let src = r#"
PI = 3.141592653589793
n0 = 6
n1 = 5
n2 = 8
f = czeros(n0, n1, n2)
g = czeros(n0, n1, n2)
for a to n0 { for b to n1 { for c to n2 {
    ph = 2.0 * PI * (2.0 * a / n0 + 1.0 * b / n1 + 3.0 * c / n2)
    f[a, b, c] = complex(cos(ph), sin(ph))
    g[a, b, c] = f[a, b, c]
} } }
fft3(g)
peak = abs(g[2, 1, 3])
leak = 0.0
for a to n0 { for b to n1 { for c to n2 {
    if abs(a - 2.0) + abs(b - 1.0) + abs(c - 3.0) > 0.5 {
        if abs(g[a, b, c]) > leak { leak = abs(g[a, b, c]) }
    }
} } }
ifft3(g)
back = 0.0
for a to n0 { for b to n1 { for c to n2 {
    if abs(g[a, b, c] - f[a, b, c]) > back { back = abs(g[a, b, c] - f[a, b, c]) }
} } }
test "one bin, height N0 N1 N2 = 240, and it inverts" {
    assert_near(peak, 240.0, 1.0e-9)
    assert_lt(leak, 1.0e-10)
    assert_lt(back, 1.0e-13)
}
"#;
    let interp = interp_lines(src);
    assert!(interp.contains("PASS "), "interpreter:\n{interp}");
    if let Some(compiled) = run_compiled(src, "fft3") {
        assert_eq!(interp, compiled, "the engines disagree on fft3");
    }
}

/// Whole-array operations the DFT code leans on, where the engines had
/// DIVERGED: the interpreter divided one array by another elementwise, and the
/// C backend refused ("cannot divide a array and a array"). Found the first
/// time a real program wrote `num / den` -- the Pade exchange-correlation
/// functional -- and compiled it.
#[test]
fn whole_array_arithmetic_agrees_across_engines() {
    let src = r#"
a = [2.0, 6.0, 9.0]
b = [1.0, 3.0, 4.5]
show a / b
show 18.0 / a
show sum(a * b)
z = czeros(2)
fill(z, 1.0 + 2.0i)
show z
r = [1.0, 2.0]
q = [3.0, 4.0]
show complex(r, q)
"#;
    let interp = interp_lines(src);
    assert_eq!(interp, "[2, 2, 2]\n[9, 3, 2]\n60.5\nz: [1+2i, 1+2i]\n[1+3i, 2+4i]\n");
    if let Some(compiled) = run_compiled(src, "arrops") {
        assert_eq!(interp, compiled);
    }
}

/// Whole-array arithmetic on COMPLEX arrays, in both engines, to the byte:
/// + - * between complex arrays, complex * real array in either order, a real
/// or complex scalar on either side, / by a scalar, `@`, and `transpose`.
/// The C runtime writes these out in real arithmetic on the packed (re, im)
/// pairs, in the interpreter's order, so there is nothing to round differently.
#[test]
fn complex_array_arithmetic_agrees_across_engines() {
    let src = r#"
a = [1.0 + 2.0i, 3.0, 0.0 - 1.0i]
b = [2.0, 0.0 + 1.0i, 4.0]
show a + b
show a - b
show a * b
show a * [2.0, 2.0, 2.0]
show [2.0, 2.0, 2.0] * a
show 2.0 * a
show a * 2.0
show a * (0.0 + 1.0i)
show (0.5 - 1.5i) * a
show a / 2.0
show a / (1.0 + 1.0i)
M = czeros(2, 2)
M[0, 0] = 1.0
M[0, 1] = 0.0 + 1.0i
M[1, 0] = 0.0 - 1.0i
M[1, 1] = 2.0
show M @ M
show transpose(M)
v = [1.0 + 0.0i, 0.0 + 1.0i]
show M @ v
show v @ M
"#;
    let interp = interp_lines(src);
    assert!(interp.contains("[3+2i, 3+1i, 4-1i]"), "a + b:\n{interp}");
    assert!(interp.contains("[[2+0i, 0+3i],\n [0-3i, 5+0i]]"), "M @ M:\n{interp}");
    if let Some(native) = run_compiled(src, "carith") {
        assert_eq!(native, interp, "complex array arithmetic diverged between the engines");
    }
}

/// `slab`/`setslab` and `take`/`put` on real and complex arrays, both engines.
#[test]
fn slab_and_take_put_agree_across_engines() {
    let src = r#"
R = czeros(3, 2)
setslab(R, 1, [1.0 + 0.0i, 0.0 + 1.0i])
show R
show slab(R, 1)
g = czeros(2, 3)
idx = [5.0, 0.0, 3.0]
put(g, idx, [1.0 + 2.0i, 3.0, 0.0 - 1.0i])
show g
show take(g, idx)
r = zeros(2, 3)
put(r, idx, [7.0, 8.0, 9.0])
show r
show take(r, [3.0, 3.0, 5.0, 0.0])
setslab(r, 0, [1.0, 1.0, 1.0])
show slab(r, 0)
cube = zeros(2, 2, 2)
cube[1, 0, 1] = 4.0
show slab(cube, 1)
"#;
    let interp = interp_lines(src);
    assert!(interp.contains("[1+2i, 3+0i, 0-1i]"), "take must undo put:\n{interp}");
    assert!(interp.contains("[[0, 4],\n [0, 0]]"), "slab of a rank-3 array is a matrix:\n{interp}");
    if let Some(native) = run_compiled(src, "slabs") {
        assert_eq!(native, interp, "slab / take / put diverged between the engines");
    }
}

/// Mixing element types where a copy would silently drop the imaginary part is
/// a compile error in C and a run-time error in the interpreter -- never a
/// quiet conversion.
#[test]
fn put_and_setslab_refuse_mixed_element_types() {
    for src in [
        "g = czeros(4)\nput(g, [0.0], [1.0])\n",
        "g = zeros(4)\nput(g, [0.0], [1.0 + 1.0i])\n",
        "R = czeros(2, 2)\nsetslab(R, 0, [1.0, 2.0])\n",
        "A = czeros(2, 2)\nB = zeros(2, 2)\nshow A @ B\n",
    ] {
        assert!(viv::compile_c(src).is_err(), "should refuse to compile:\n{src}");
        assert!(viv::interp::run(src).is_err(), "should refuse to interpret:\n{src}");
    }
}

/// A parameter that is only ever handed to `transpose` still takes its type
/// from the call site: `transpose` accepts either element type, so "real
/// array" is a default, not evidence.
#[test]
fn transpose_parameter_takes_its_element_type_from_the_call_site() {
    let src = r#"
def flip(m) {
    return transpose(m)
}
A = czeros(2, 2)
A[0, 1] = 0.0 + 1.0i
show flip(A)
B = zeros(2, 2)
B[0, 1] = 5.0
show transpose(B)
"#;
    let interp = interp_lines(src);
    assert!(interp.contains("[[0+0i, 0+0i],\n [0+1i, 0+0i]]"), "{interp}");
    if let Some(native) = run_compiled(src, "flip") {
        assert_eq!(native, interp);
    }
}

/// Every FFT method -- radix-2, mixed radix (with its dedicated radix-3 and
/// radix-5 butterflies and the generic radix-7 one) and Bluestein -- against a
/// naive O(N^2) transform and a round trip, then the low-order BITS of a
/// checksum across engines. `t`, `u`, `v` are digits 1-6, 6-11 and 11-16 of
/// the checksum: agreement in `v` is agreement to the last bit.
#[test]
fn every_fft_method_is_correct_and_bit_identical_across_engines() {
    let src = r#"
def check_size(n) {
    x = czeros(n)
    for j to n {
        x[j] = complex(sin(0.37 * j) + 0.01 * j, cos(1.3 * j * j / n))
    }
    X = czeros(n)
    for j to n {
        X[j] = x[j]
    }
    fft(X)
    worst = 0.0
    scale = 0.0
    for k to n {
        acc = complex(0.0, 0.0)
        for j to n {
            ang = -2.0 * 3.141592653589793 * mod(k * j, n) / n
            acc = acc + x[j] * complex(cos(ang), sin(ang))
        }
        worst = max(worst, abs(X[k] - acc))
        scale = max(scale, abs(acc))
    }
    s = 0.0
    for k to n {
        s = s + re(X[k]) / (k + 1.0) + im(X[k]) / (k + 2.0)
    }
    ifft(X)
    back = 0.0
    for j to n {
        back = max(back, abs(X[j] - x[j]))
    }
    ok = 0.0
    if worst / scale < 1.0e-13 and back < 1.0e-13 { ok = 1.0 }
    t = s - round(s)
    u = t * 1.0e5 - round(t * 1.0e5)
    v = u * 1.0e5 - round(u * 1.0e5)
    print(n)
    print(ok)
    print(t)
    print(u)
    print(v)
    return 0.0
}
SIZES = [2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 11.0, 16.0, 18.0, 25.0, 30.0, 36.0, 49.0, 64.0, 97.0, 100.0, 210.0]
for i to len(SIZES) {
    check_size(SIZES[i])
}
"#;
    let interp = interp_lines(src);
    let rows: Vec<&str> = interp.lines().collect();
    assert_eq!(rows.len(), 17 * 5);
    for row in rows.chunks(5) {
        assert_eq!(row[1], "1", "fft wrong (vs the naive transform, or the round trip) at n = {}", row[0]);
    }
    if let Some(native) = run_compiled(src, "fftall") {
        assert_eq!(native, interp, "an FFT method is not bit-identical between the engines");
    }
}

/// The zero-line skip in fft3 must not change a result: a box holding a small
/// sphere (most lines empty) transforms to the same numbers as the separable
/// definition says, there and back, in both engines.
#[test]
fn fft3_of_a_mostly_empty_box() {
    let src = r#"
n = 6.0
A = czeros(n, n, n)
A[0, 0, 0] = 1.0 + 0.5i
A[1, 0, 5] = 0.0 - 2.0i
A[5, 1, 0] = 3.0
ifft3(A)
p = A[2, 3, 4]
show re(p) * 216.0
show im(p) * 216.0
fft3(A)
worst = abs(A[0, 0, 0] - (1.0 + 0.5i))
worst = max(worst, abs(A[1, 0, 5] - (0.0 - 2.0i)))
worst = max(worst, abs(A[5, 1, 0] - 3.0))
worst = max(worst, abs(A[2, 2, 2]))
back = 0.0
if worst < 1.0e-14 { back = 1.0 }
show back
"#;
    let interp = interp_lines(src);
    // x[r] = sum_G A[G] exp(+2 pi i G.r / n) / n^3, r = (2, 3, 4), by hand:
    //   G=(0,0,0): 1 + 0.5i
    //   G=(1,0,5): phase 2 pi (2 + 20)/6 = 2 pi * 22/6 -> exp(i 4 pi / 3) * (-2i)
    //   G=(5,1,0): phase 2 pi (10 + 3)/6 -> exp(i pi / 3) * 3
    let (c1, s1) = ((4.0 * std::f64::consts::PI / 3.0).cos(), (4.0 * std::f64::consts::PI / 3.0).sin());
    let (c2, s2) = ((std::f64::consts::PI / 3.0).cos(), (std::f64::consts::PI / 3.0).sin());
    let want_re = 1.0 + 2.0 * s1 + 3.0 * c2;
    let want_im = 0.5 - 2.0 * c1 + 3.0 * s2;
    let got: Vec<f64> = interp.lines().take(2).map(|l| l.trim().parse().unwrap()).collect();
    assert!((got[0] - want_re).abs() < 1e-5 && (got[1] - want_im).abs() < 1e-5, "ifft3 value wrong: {got:?} vs ({want_re}, {want_im})");
    assert_eq!(interp.lines().nth(2), Some("back: 1"), "the round trip must return the box:\n{interp}");
    if let Some(native) = run_compiled(src, "fft3sparse") {
        assert_eq!(native, interp);
    }
}

/// Generated C turns floating-point contraction OFF for the whole file. With it
/// on, clang on arm64 fuses `a * b - c` into one multiply-add that rounds once
/// where the interpreter rounds twice. The expression below is built so the
/// two differ: a * a is 1 + 2^-26 + 2^-54 exactly, which ROUNDS to 1 + 2^-26,
/// so the separate multiply-then-subtract gives 0 and a fused one gives 2^-54.
#[test]
fn compiled_arithmetic_is_not_contracted_into_fma() {
    let src = "a = 1.0 + pow(2.0, -27.0)\nc = 1.0 + pow(2.0, -26.0)\nd = (a * a - c) * pow(2.0, 54.0)\nshow d\n";
    assert!(emit_c(src).starts_with("#pragma STDC FP_CONTRACT OFF\n"));
    let interp = interp_lines(src);
    assert_eq!(interp, "d: 0\n");
    if let Some(native) = run_compiled(src, "fma") {
        assert_eq!(native, interp, "the compiled program fused a multiply-add");
    }
}

/// The array arena. A function that can allocate frees, on every way out,
/// what it allocated -- except the array it returns. A function that cannot
/// allocate compiles exactly as before, with no arena code at all.
#[test]
fn array_arena_frees_function_temporaries_and_keeps_what_escapes() {
    let src = r#"
def scalar_only(x) {
    return x * x + 1.0
}
def makes(n) {
    a = zeros(n)
    b = a + 1.0
    return b * 2.0
}
def reduces(a, b) {
    return sum(a * b)
}
def passes_through(a) {
    t = a + 1.0
    return a
}
def early(a, flag) {
    t = a * 3.0
    if flag > 0.5 {
        return t
    }
    return t + t
}
def mutates(a) {
    t = a + 10.0
    a[0] = t[1]
}
def nested(n) {
    return makes(n) + makes(n)
}
g = makes(4.0)
show g
show reduces(g, g)
show passes_through(g)
show early(g, 1.0)
show early(g, 0.0)
mutates(g)
show g
total = 0.0
for i to 2000 {
    w = nested(500.0)
    total = total + w[7] + scalar_only(2.0)
}
show total
"#;
    let c = emit_c(src);
    let body_of = |name: &str| -> &str {
        let start = c.rfind(&format!("{name}(")).expect("function body present");
        &c[start..start + c[start..].find("\n}\n").unwrap()]
    };
    assert!(!body_of("double scalar_only").contains("viv_"), "a scalar function must not touch the arena:\n{}", body_of("double scalar_only"));
    assert!(body_of("viv_arr makes").contains("viv_arena_release(viv_mark, viv_ret.data)"));
    assert!(body_of("double reduces").contains("viv_arena_release(viv_mark, (const double*)0)"));
    assert!(body_of("void mutates").contains("viv_arena_release(viv_mark, (const double*)0)"));
    assert_eq!(body_of("viv_arr early").matches("viv_arena_release").count(), 2, "both returns release");
    let interp = interp_lines(src);
    assert!(interp.contains("total: 18000"), "{interp}");
    if let Some(native) = run_compiled(src, "arena") {
        assert_eq!(native, interp, "freeing function temporaries changed a result");
    }
}

/// A scalar-only program's C must not mention the arena (or any array runtime).
#[test]
fn scalar_programs_carry_no_arena() {
    let c = emit_c("def f(x) {\n    return x * 2.0\n}\nprint(f(3.0))\n");
    assert!(!c.contains("viv_arena"), "{c}");
}

/// `max` / `min` with an array operand are elementwise, scalar on either side,
/// in both engines -- and a parameter that is only handed to `max` takes its
/// type from the call site. (Found by copper: a mixed density dips below zero,
/// `log` of it is NaN, and the floor `max(rho, 1e-30)` had no spelling.)
#[test]
fn min_max_are_elementwise_over_arrays() {
    let src = r#"
a = [-1.0, 0.5, 3.0]
b = [0.0, 1.0, 2.0]
show max(a, 0.0)
show max(0.25, a)
show min(a, 1.0)
show max(a, b)
show min(a, b)
show max(2.0, 3.0)
def floor_it(x) {
    return max(x, 1.0e-30)
}
show floor_it(a)
"#;
    let interp = interp_lines(src);
    assert_eq!(interp, "[0, 0.5, 3]\n[0.25, 0.5, 3]\n[-1, 0.5, 1]\n[0, 1, 3]\n[-1, 0.5, 2]\n3\n[1e-30, 0.5, 3]\n");
    if let Some(native) = run_compiled(src, "minmax") {
        assert_eq!(native, interp);
    }
}

/// `Bool` is a real type with ONE semantics. It can be stored, passed,
/// returned, combined and shown, and the interpreter and the compiled program
/// print the same thing -- `true` / `false`, not 1 / 0.
#[test]
fn booleans_are_values_in_both_engines() {
    let src = r#"
found = false
show found
n = 7.0
small = n < 10.0
show small
both = small and not found
show both
show small == both
show small != found
def on_boundary(x: Num, y: Num, tol: Num) -> Bool {
    return x < tol or x > 1.0 - tol or y < tol or y > 1.0 - tol
}
show on_boundary(0.0, 0.5, 1.0e-9)
show on_boundary(0.5, 0.5, 1.0e-9)
def first_negative(a) {
    hit = false
    where = -1.0
    for i to len(a) {
        if not hit and a[i] < 0.0 {
            hit = true
            where = i
        }
    }
    return where
}
show first_negative([3.0, 1.0, -2.0, -5.0])
done = false
count = 0.0
while not done {
    count = count + 1.0
    done = count >= 4.0
}
show count
test "booleans are values" {
    assert_true(done)
    assert_true(on_boundary(1.0, 0.2, 1.0e-9))
}
"#;
    let interp = interp_lines(src);
    assert_eq!(
        interp,
        "found: false\nsmall: true\nboth: true\ntrue\ntrue\ntrue\nfalse\n2\ncount: 4\nPASS booleans are values\n"
    );
    if let Some(native) = run_compiled(src, "bools") {
        assert_eq!(native, interp, "a boolean printed differently in the two engines");
    }
}

/// ... and it is NOT a number. Every misuse is refused by the interpreter and
/// by the compiler alike: before `Bool` existed the interpreter accepted most of
/// these (a comparison was the number 1.0) and the C backend refused them.
#[test]
fn booleans_are_not_numbers_in_either_engine() {
    for src in [
        "x = true + 1.0\n",
        "if 1.0 {\n    show 2.0\n}\n",
        "c = 3.0\nwhile c {\n    c = c - 1.0\n}\n",
        "x = true\ny = x < false\n",
        // a bare `print(1.0 < 2.0)` writes `true`; a format is for numbers
        "print(fmt(1.0 < 2.0))\n",
        "x = 1.0 and true\n",
        "f = 2.0 > 1.0\nshow f * 3.0\n",
    ] {
        assert!(viv::compile_c(src).is_err(), "the compiler should refuse:\n{src}");
        assert!(viv::interp::run(src).is_err(), "the interpreter should refuse:\n{src}");
    }
}

/// A Bool is exactly 1.0 or 0.0 inside a C double, and the literal has its own
/// IR node: `x = true` and `x = 1.0` are different programs with different
/// content addresses.
#[test]
fn bool_literal_has_its_own_content_address() {
    let c = emit_c("x = true\nshow x\n");
    assert!(c.contains("x = 1.0;") && c.contains("viv_show_bool(\"x\", x)"), "{c}");
    // a function's address covers its body, so the two defs differ only in that literal
    let t = viv::hash_report("def f() {\n    return true\n}\n").unwrap();
    let one = viv::hash_report("def f() {\n    x = 1.0\n    return x > 0.0\n}\n").unwrap();
    assert_ne!(t, one);
    let a = viv::hash_report("def g() {\n    flag = true\n    return flag\n}\n").unwrap();
    let b = viv::hash_report("def g() {\n    flag = false\n    return flag\n}\n").unwrap();
    assert_ne!(a, b, "true and false must not share a content address");
}

/// A chart's title and axis labels are part of the program: a plot with no
/// `figure(title, xlabel, ylabel)` declared before it is refused -- by the
/// interpreter and by the compiler, since both parse through one function.
#[test]
fn a_plot_must_declare_its_figure() {
    for call in ["plot(x, x, \"line\")", "scatter(x, x, \"pts\")", "plot3d(x, x, x, \"curve\")", "quiver(x, x, x, x, \"field\")"] {
        let bare = format!("x = linspace(0.0, 1.0, 5)\n{call}\n");
        let c = viv::compile_c(&bare).expect_err("the compiler should refuse an undeclared chart");
        assert!(c.contains("needs a `figure(title, xlabel, ylabel)` before it"), "{c}");
        assert!(viv::interp::run(&bare).is_err(), "the interpreter should refuse:\n{bare}");
        let declared = format!("x = linspace(0.0, 1.0, 5)\nfigure(\"A chart\", \"x\", \"y\")\n{call}\n");
        assert!(viv::compile_c(&declared).is_ok(), "should compile:\n{declared}");
        assert!(viv::interp::run(&declared).is_ok(), "should run:\n{declared}");
    }
    // declared too late is still undeclared
    assert!(viv::compile_c("x = linspace(0.0, 1.0, 5)\nplot(x, x, \"line\")\nfigure(\"late\", \"x\", \"y\")\n").is_err());
    // a drawing helper may rely on a figure the program declares at the top level ...
    let helper = "def draw(x) {\n    plot(x, x, \"line\")\n}\nfigure(\"A chart\", \"x\", \"y\")\ndraw(linspace(0.0, 1.0, 5))\n";
    assert!(viv::compile_c(helper).is_ok() && viv::interp::run(helper).is_ok());
    // ... or declare its own; with neither, it is refused
    let own = "def draw(x) {\n    figure(\"Mine\", \"x\", \"y\")\n    plot(x, x, \"line\")\n}\ndraw(linspace(0.0, 1.0, 5))\n";
    assert!(viv::compile_c(own).is_ok() && viv::interp::run(own).is_ok());
    let neither = "def draw(x) {\n    plot(x, x, \"line\")\n}\ndraw(linspace(0.0, 1.0, 5))\n";
    assert!(viv::compile_c(neither).is_err() && viv::interp::run(neither).is_err());
}

/// A content address must not depend on WHERE the code sits. A lifted closure
/// used to be named after its byte offset (`viv_lam_1807`), and that name is
/// hashed into the enclosing function -- so inserting an unrelated line above
/// `area` changed `area`'s address. (Found by the H2 ladder, whose whole point
/// is that an untouched function keeps its address from rung to rung.)
#[test]
fn a_function_with_a_closure_keeps_its_address_when_code_moves() {
    let f = "def area(T) {\n    return simpson(fn(t) -> exp(-T * t * t), 0.0, 1.0, 60)\n}\n";
    let alone = viv::hash_report(f).unwrap();
    let moved = viv::hash_report(&format!("# a comment, and another function, above it\ndef other(x) {{\n    return x + 1.0\n}}\n{f}")).unwrap();
    let address_of = |report: &str| report.lines().find(|l| l.contains("def area(")).map(|l| l.split_whitespace().take(3).collect::<Vec<_>>().join(" ")).unwrap();
    assert_eq!(address_of(&alone), address_of(&moved), "`area` changed address because code was inserted above it");
}

/// The same closure written twice is ONE lifted function (same content, same
/// name), and the program still compiles and agrees with the interpreter.
#[test]
fn identical_closures_lift_to_one_function() {
    let src = "a = simpson(fn(t) -> t * t, 0.0, 1.0, 20)\nb = simpson(fn(t) -> t * t, 0.0, 3.0, 20)\nshow a\nshow b\n";
    let c = emit_c(src);
    // one distinct lifted name, defined once
    let mut names: Vec<&str> = c.match_indices("viv_lam_").map(|(k, _)| &c[k..k + 16]).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), 1, "expected one lifted closure, got {names:?}");
    assert_eq!(c.lines().filter(|l| l.contains(names[0]) && l.trim_end().ends_with('{')).count(), 1, "defined more than once:\n{c}");
    let interp = interp_lines(src);
    assert_eq!(interp, "a: 0.333333\nb: 9\n");
    if let Some(native) = run_compiled(src, "twolam") {
        assert_eq!(native, interp);
    }
}

/// The two builtins the PDE solvers lean on, in both engines: `tridiag_solve`
/// (the Thomas algorithm, checked here against the residual of the dense system
/// it represents) and `roll` (the circular shift that makes a periodic stencil
/// one whole-array line).
#[test]
fn tridiag_solve_and_roll_agree_across_engines() {
    let src = r#"# tridiag_solve against a dense solve of the same system, and roll against its definition
n = 7
lo = zeros(n)
di = zeros(n)
up = zeros(n)
rhs = zeros(n)
A = zeros(n, n)
for i to n {
    di[i] = 4.0 + 0.3 * i
    rhs[i] = sin(1.0 + i)
    A[i, i] = di[i]
    if i > 0.0 {
        lo[i] = -1.0 - 0.1 * i
        A[i, i - 1.0] = lo[i]
    }
    if i < n - 1.0 {
        up[i] = -1.5 + 0.2 * i
        A[i, i + 1.0] = up[i]
    }
}
x = tridiag_solve(lo, di, up, rhs)
resid = A @ x - rhs
worst = 0.0
for i to n {
    worst = max(worst, abs(resid[i]))
}
show x
test "tridiag_solve solves the system: |A x - d| is rounding" {
    assert_lt(worst, 1.0e-15)
}
u = [1.0, 2.0, 3.0, 4.0, 5.0]
show roll(u, 1.0)
show roll(u, -1.0)
show roll(u, 7.0)
M = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
show roll(M, 1.0, 1.0)
show roll(M, 1.0, 0.0)
lap = roll(u, 1.0) - 2.0 * u + roll(u, -1.0)
show lap
"#;
    let interp = interp_lines(src);
    assert!(interp.contains("PASS tridiag_solve solves the system"), "{interp}");
    assert!(interp.contains("[5, 1, 2, 3, 4]") && interp.contains("[2, 3, 4, 5, 1]") && interp.contains("[4, 5, 1, 2, 3]"), "roll: {interp}");
    assert!(interp.contains("lap: [5, 0, 0, 0, -5]"), "the periodic second difference of a ramp: {interp}");
    if let Some(native) = run_compiled(src, "pdebuiltins") {
        assert_eq!(native, interp);
    }
    // a singular (not diagonally dominant) system is refused, not divided through
    assert!(viv::interp::run("z = zeros(2)
x = tridiag_solve(z, z, z, [1.0, 1.0])
").is_err());
    // roll(A, k) on a matrix needs its axis
    assert!(viv::interp::run("A = zeros(2, 2)
B = roll(A, 1.0)
").is_err());
}

/// Unary minus on an array -- real or complex, a literal, a variable or a
/// parameter -- in both engines. (A PDE solver's first line is usually
/// `exp(-(x - x0) * (x - x0))`, and it used to be "expected a number for
/// negation, got an array".) And a negative zero prints as `0` from the compiled
/// program too: the interpreter always printed it so; C's %g printed `-0`.
#[test]
fn unary_minus_on_arrays_and_negative_zero_prints_alike() {
    let src = "a = [1.0, -2.0, 0.0]\nshow -a\nshow exp(-a * a)\nz = [1.0 + 2.0i, 0.0 - 1.0i]\nshow -z\ndef neg(v) {\n    return -v\n}\nshow neg(a)\nzero = 0.0\nshow -zero\nprint(-zero)\n";
    let interp = interp_lines(src);
    assert_eq!(interp, "[-1, 2, 0]\n[0.367879, 0.0183156, 1]\n[-1-2i, 0+1i]\n[-1, 2, 0]\n0\n0\n");
    if let Some(native) = run_compiled(src, "negarr") {
        assert_eq!(native, interp);
    }
}

#[test]
fn pow_maps_over_an_array_with_a_scalar_exponent() {
    // pow(array, s) is elementwise in both engines; the base's type is inferred
    // through a function parameter, and a scalar base still goes to libm.
    let src = r#"
def cube_root(a) {
    return pow(a, 1.0 / 3.0)
}
a = [1.0, 8.0, 27.0, 2.0]
c = cube_root(a)
show c[1]
show c[2]
show c[3]
show pow(2.0, 10.0)
s = pow(a, 2.0)
show sum(s)
"#;
    let out = interp_lines(src);
    assert_eq!(out, "2\n3\n1.25992\n1024\n798\n");
    if let Some(native) = run_compiled(src, "pow_arr") {
        assert_eq!(native, out, "pow over an array printed differently in the two engines");
    }
}

#[test]
fn max_and_min_of_one_array_are_reductions() {
    // max(a) / min(a) reduce; max(a, s) stays elementwise. A parameter reduced
    // this way is inferred to be an array.
    let src = r#"
def max_norm(d) {
    return max(abs(d))
}
a = [[1.0, -7.5], [3.0, 2.0]]
show max_norm(a)
show max(a)
show min(a)
f = max(a, 0.0)
show min(f)
"#;
    let out = interp_lines(src);
    assert_eq!(out, "7.5\n3\n-7.5\n0\n");
    if let Some(native) = run_compiled(src, "extent") {
        assert_eq!(native, out, "max/min reductions printed differently in the two engines");
    }
}

/// `cos(theta)` does not make `theta` a number: every one-argument math function
/// also maps over an array. The signature inference used to push `Num` backward
/// into the parameter on first sight, and then refused the array the function
/// was written for -- in the COMPILER only; the interpreter has no inference to
/// get wrong. Found by running the PDE solvers through both engines.
#[test]
fn math_on_an_untyped_parameter_does_not_make_it_a_number() {
    let src = r#"
N = 4
K = linspace(0.0, 3.0, N)
def phase(theta) {
    return complex(cos(theta), sin(theta))
}
def turn(u, dt) {
    v = complex(u, zeros(N))
    v = v * phase(K * dt)
    return re(v)
}
def damp(u) {
    return exp(-1.0 * u)
}
w = turn(K, 0.5)
show w[2]
d = damp(K)
show d[1]
"#;
    let out = interp_lines(src);
    assert_eq!(out, "1.0806\n0.367879\n");
    if let Some(native) = run_compiled(src, "overloaded_math") {
        assert_eq!(native, out);
    }
}

/// In a function an assignment makes a LOCAL, shadowing a global of the same
/// name -- including during the passes of signature inference in which the
/// local's type is not known yet. It used to fall through to the global then:
/// here the global `S` is a matrix, so `half` was inferred to return an array
/// and the (correct) checked lowering refused it. Whether it bit depended on
/// how early the callee's return type arrived, i.e. on definition order.
#[test]
fn an_untyped_local_shadows_a_global_during_inference() {
    let src = r#"
S = zeros(2, 2)
def late(x) {
    y = exp(x)
    return y
}
def half(a) {
    S = late(a)
    return 0.5 * S
}
S[0, 0] = 4.0
show half(0.0)
show S[0, 0]
"#;
    let out = interp_lines(src);
    assert_eq!(out, "0.5\n4\n");
    if let Some(native) = run_compiled(src, "shadow") {
        assert_eq!(native, out);
    }
}

/// A figure's axis window prints the same in both engines. It did not: the
/// interpreter wrote Rust's `{:e}` (`3.141592653589793e0`) and the C runtime
/// `%e` (`3.141593e+00`). Found by running the feed's posts through both.
#[test]
fn a_figure_window_prints_alike_in_both_engines() {
    let src = r#"
x = linspace(0.0, 1.0, 3)
figure("windowed", "x", "y", 0.0, 3.141592653589793, -0.5, 1.5)
plot(x, x, "line")
"#;
    let out = interp_lines(src);
    assert!(out.starts_with("figure: windowed ; x ; y ; 0 3.14159 -0.5 1.5\n"), "{out}");
    if let Some(native) = run_compiled(src, "figwin") {
        assert_eq!(native, out);
    }
}

/// `scatter` carries full-precision data, and both engines must print the same
/// bytes: C's `%.17e` writes `e+00` where Rust's `{:.17e}` writes `e0`.
#[test]
fn scatter_prints_the_same_bytes_in_both_engines() {
    let src = r#"
x = [2.8, 1.0e-15, -3.0e20, 0.0]
figure("points", "x", "y")
scatter(x, x, "pts")
"#;
    let out = interp_lines(src);
    assert!(out.contains("2.79999999999999982e+00"), "{out}");
    assert!(out.contains("1.00000000000000008e-15"), "{out}");
    assert!(out.contains("-3.00000000000000000e+20"), "{out}");
    if let Some(native) = run_compiled(src, "scatter_fmt") {
        assert_eq!(native, out);
    }
}

/// Text in a string literal is UTF-8 and reaches the output intact in both
/// engines. The lexer used to push each BYTE as a char (mojibake in the
/// interpreter), and the C backend wrote Rust's `\u{80}` escapes, which are not
/// C ("universal character name refers to a control character").
#[test]
fn non_ascii_text_survives_both_engines() {
    let src = "x = [0.0, 1.0]\nfigure(\"Schr\u{f6}dinger \u{2014} \u{3c0}\", \"x\", \"what?? a \\\"quote\\\"\")\nplot(x, x, \"\u{3c8}\")\n";
    let out = interp_lines(src);
    assert!(out.starts_with("figure: Schr\u{f6}dinger \u{2014} \u{3c0} ; x ; what?? a \"quote\"\n"), "{out}");
    assert!(out.contains("plot: \u{3c8} ;"), "{out}");
    if let Some(native) = run_compiled(src, "utf8_text") {
        assert_eq!(native, out);
    }
}

/// The animation verbs print one record each, the same bytes from both engines,
/// and a program's own function of the same name still wins.
#[test]
fn animation_records_match_in_both_engines() {
    let src = r#"
x = linspace(0.0, 1.0, 3)
stage("A line becomes a parabola", 0.0, 1.0, -0.5, 1.0)
axes("x", "y")
curve("f", x, x)
hline("zero", 0.0)
label("name", "$y = x$", 0.8, 0.9)
together()
draw("f", 1.5)
fade_in("name", 0.5)
apart()
say("now square it")
morph("f", x, x * x, 2.0)
say("the value at the end is", x[2] * x[2])
move_to("name", 0.5, 0.1, 1.0)
fade_out("zero", 0.25)
wait(1.0)
"#;
    let out = interp_lines(src);
    assert_eq!(
        out,
        "stage: A line becomes a parabola ; 0 1 -0.5 1\n\
         axes: x ; y\n\
         curve: f ; [0, 0.5, 1] ; [0, 0.5, 1]\n\
         hline: zero ; 0\n\
         label: name ; $y = x$ ; 0.8 0.9\n\
         together:\n\
         draw: f ; 1.5\n\
         fade_in: name ; 0.5\n\
         apart:\n\
         say: now square it\n\
         morph: f ; [0, 0.5, 1] ; [0, 0.25, 1] ; 2\n\
         say: the value at the end is ; 1\n\
         move_to: name ; 0.5 0.1 1\n\
         fade_out: zero ; 0.25\n\
         wait: 1\n"
    );
    if let Some(native) = run_compiled(src, "anim") {
        assert_eq!(native, out, "an animation record printed differently in the two engines");
    }
}

#[test]
fn a_users_function_named_like_an_animation_verb_wins() {
    let src = "def wait(n) {\n    return n * 2.0\n}\nshow wait(4.0)\n";
    let out = interp_lines(src);
    assert_eq!(out, "8\n");
    if let Some(native) = run_compiled(src, "anim_shadow") {
        assert_eq!(native, out);
    }
}

#[test]
fn animation_verbs_refuse_the_wrong_shape_in_both_engines() {
    for src in [
        "draw(\"f\")\n",                              // a missing number
        "x = [0.0, 1.0]\ncurve(x, x, x)\n",            // the id is not a string literal
        "curve(\"f\", 1.0, 2.0)\n",                    // numbers where arrays go
        "say(\"a ; b\")\n",                            // the record separator inside text
    ] {
        assert!(viv::compile_c(src).is_err(), "the compiler should refuse:\n{src}");
        assert!(viv::interp::run(src).is_err(), "the interpreter should refuse:\n{src}");
    }
}

/// `--set NAME=value` changes one top-level constant and nothing else: the
/// program is the same by the content address of every function. Substitution
/// is textual and happens before parsing, so it is the same for both engines.
#[test]
fn set_changes_a_top_level_constant_and_keeps_every_address() {
    let src = "N = 200    # points\nSTEPS = 10\ndef f(x) {\n    return x * x\n}\nx = linspace(0.0, 1.0, N)\nshow len(x)\nshow f(STEPS)\n";
    let big = viv::apply_sets(src, &[("N".to_string(), "2000".to_string()), ("STEPS".to_string(), "1e3".to_string())]).unwrap();
    assert!(big.contains("N = 2000    # points\n"), "{big}");
    assert!(big.contains("STEPS = 1e3\n"), "{big}");
    assert_eq!(interp_lines(&big), "2000\n1000000\n");
    let hash = |s: &str| {
        let ir = viv::build_ir(s, knotkit::span::SourceId(0)).unwrap();
        ir.roots().iter().filter_map(|r| match &ir.get(*r).kind { viv::lower::Viv::Func { name, .. } => Some((name.clone(), r.short())), _ => None }).collect::<Vec<_>>()
    };
    assert_eq!(hash(src), hash(&big), "a def's content address must not move with a --set");
    assert!(viv::apply_sets(src, &[("M".to_string(), "1".to_string())]).is_err(), "a name with no top-level line is an error, not a no-op");
    assert!(viv::apply_sets(src, &[("N".to_string(), "many".to_string())]).is_err());
    // a string constant takes a double-quoted string, and only a string constant does
    let cls = "CLASS = \"S\"    # problem size\nprint(CLASS)\n";
    let w = viv::apply_sets(cls, &[("CLASS".to_string(), "\"W\"".to_string())]).unwrap();
    assert_eq!(interp_lines(&w), "W\n");
    assert!(viv::apply_sets(src, &[("N".to_string(), "\"W\"".to_string())]).is_err(), "a string never replaces a number");
    assert!(viv::apply_sets(cls, &[("CLASS".to_string(), "7".to_string())]).is_err(), "a number never replaces a string");
    assert!(viv::apply_sets(cls, &[("CLASS".to_string(), "\"a\"b\"".to_string())]).is_err());
}

/// `histogram` counts into equal bins, the last one closed on the right, and
/// `bars` draws them; both engines print the same record.
#[test]
fn histogram_and_bars_agree_in_both_engines() {
    let src = r#"
v = [0.0, 0.1, 0.5, 0.99, 1.0, 1.5, 2.0, 2.0, 7.0, -1.0]
h = histogram(v, 0.0, 2.0, 4)
show h
show sum(h)
centers = linspace(0.25, 1.75, 4)
figure("counts", "value", "count")
bars(centers, h, "histogram")
"#;
    let out = interp_lines(src);
    assert_eq!(out, "h: [2, 2, 1, 3]\n8\nfigure: counts ; value ; count\nbars: histogram ; [0.25, 0.75, 1.25, 1.75] ; [2, 2, 1, 3]\n");
    if let Some(native) = run_compiled(src, "hist") {
        assert_eq!(native, out);
    }
}

/// A ladder is a program of rungs. Run, it is its top rung, in both engines; a
/// def in a later rung replaces the earlier one; `viv ladder` runs every rung
/// and holds it to the rules.
#[test]
fn bit_operations_agree_in_both_engines() {
    // The arithmetic a gate, an adder and an instruction decoder are made of.
    // Both engines wrap the same way, so a program that builds a CPU out of
    // these gets the same answer interpreted and compiled.
    let src = r#"
show band(12.0, 10.0)
show bor(12.0, 10.0)
show bxor(12.0, 10.0)
show bnot(0.0)
show shl(1.0, 31.0)
show shr(4294967295.0, 28.0)
show sar(i32(-8.0), 1.0)
show i32(4294967295.0)
show u32(-1.0)
show u32(4294967296.0)
show u32(2.7)
show u32(-2.7)
show shl(1.0, 32.0)
show shr(1.0, -1.0)
show sar(i32(-1.0), 99.0)
show i32(sar(i32(-1.0), 99.0))
test "xor is addition without the carry, and and is the carry" {
    a = 12.0
    b = 10.0
    assert_near(bxor(a, b), 6.0, 0.0)
    assert_near(shl(band(a, b), 1.0), 16.0, 0.0)
    assert_near(bxor(a, b) + shl(band(a, b), 1.0), a + b, 0.0)
}
test "a byte round-trips through its bits" {
    v = 0.0
    for i to 8 {
        bit = band(shr(213.0, i), 1.0)
        v = bor(v, shl(bit, i))
    }
    assert_near(v, 213.0, 0.0)
}
"#;
    let out = interp_lines(src);
    assert_eq!(
        out,
        "8\n14\n6\n4294967295\n2147483648\n15\n4294967292\n-1\n4294967295\n0\n2\n4294967294\n0\n0\n4294967295\n-1\nPASS xor is addition without the carry, and and is the carry\nPASS a byte round-trips through its bits\n"
    );
    if let Some(native) = run_compiled(src, "bits") {
        assert_eq!(native, out, "the C backend must wrap exactly as the interpreter does");
    }
}

#[test]
fn a_fenced_block_is_text_not_program() {
    // `###` on a line of its own opens a block the lexer skips to the next one:
    // a post's title and prose live there, with no `#` in front of every line.
    let src = r#"###
post: A fenced header
date: 2026-09-22
tags: lexer

Prose with no prefix. Braces { } and "quotes" and even `code` are just text,
and so is a line that would otherwise be a syntax error: 3 +

## A heading
###

x = 2.0
show x * 3.0
test "the program after the fence runs" {
    assert_near(x, 2.0, 1.0e-12)
}
"#;
    assert_eq!(interp_lines(src), "6
PASS the program after the fence runs
");
    if let Some(native) = run_compiled(src, "fenced") {
        assert_eq!(native, interp_lines(src));
    }
    // an unclosed fence swallows the rest of the file rather than half-parsing it
    let unclosed = "###
post: no end
x = 1.0
";
    assert_eq!(interp_lines(unclosed), "");
    // a fence anywhere: prose may sit between two parts of a program, and the
    // block ends at the next `###`
    let between = "x = 1.0
###
Why the next line doubles it, in plain words.
###
show x * 2.0
";
    assert_eq!(interp_lines(between), "2
");
    // a line that merely starts with # is an ordinary comment, fence or not
    assert_eq!(interp_lines("### not a fence, there is text after it
show 1.0
"), "1
");
}

#[test]
fn a_ladder_runs_as_its_top_rung_and_reports_rung_by_rung() {
    let src = r#"
H = 1.0e-3
rung "A forward difference" {
    #| the secant from x to x + h
    def slope(x) {
        return (sin(x + H) - sin(x)) / H
    }
    test "to about h" {
        assert_near(slope(0.0), 1.0, 1.0e-3)
    }
}
rung "Centered: second order" {
    def slope(x) {
        return (sin(x + H) - sin(x - H)) / (2.0 * H)
    }
    test "to h squared" {
        assert_near(slope(0.0), 1.0, 1.0e-6)
    }
}
show slope(0.0)
"#;
    let out = interp_lines(src);
    assert_eq!(out, "PASS to about h\nPASS to h squared\n1\n");
    if let Some(native) = run_compiled(src, "ladder") {
        assert_eq!(native, out);
    }
    let json = viv::ladder::report(src, &mut |t| viv::interp::run(t)).unwrap();
    assert!(json.contains(r#""n":1,"title":"A forward difference""#), "{json}");
    assert!(json.contains(r#""added":[{"name":"slope","address":"#), "{json}");
    assert!(json.contains(r#""edited":[{"name":"slope","from":"#), "{json}");
    assert!(!json.contains(r#""broken":[""#), "no rule is broken: {json}");
    // a rung that drops a test, or adds none, breaks the rules
    let bad = src.replace("    test \"to h squared\" {\n        assert_near(slope(0.0), 1.0, 1.0e-6)\n    }\n", "");
    let json = viv::ladder::report(&bad, &mut |t| viv::interp::run(t)).unwrap();
    assert!(json.contains("adds no new test"), "{json}");
    assert!(viv::ladder::report("x = 1.0\n", &mut |t| viv::interp::run(t)).is_err());
}

/// Imports nest up to `MAX_USE_DEPTH` deep: a chain of modules `m0`
/// (which imports `m1`, which imports `m2`, ...) of that length expands, and one
/// more is refused with a message that names the limit.
#[test]
fn use_lines_nest_up_to_the_limit() {
    fn chain(depth: usize) -> impl FnMut(&str) -> Result<String, String> {
        move |name: &str| {
            let k: usize = name.trim_start_matches('m').parse().map_err(|_| format!("no module {name}"))?;
            if k + 1 < depth {
                Ok(format!("from m{} import f{}\ndef f{k}(x) {{\n    return f{}(x) + 1.0\n}}\n", k + 1, k + 1, k + 1))
            }
            else {
                Ok(format!("def f{k}(x) {{\n    return x\n}}\n"))
            }
        }
    }
    let src = "from m0 import f0\nshow f0(0.0)\n";
    let out = viv::expand_uses(src, &mut chain(viv::MAX_USE_DEPTH)).expect("a chain at the limit expands");
    let ran = viv::interp::run(&out).expect("runs");
    assert!(ran.contains(&format!("{}", viv::MAX_USE_DEPTH - 1)), "f0(0) counts the chain, got:\n{ran}");
    let e = viv::expand_uses(src, &mut chain(viv::MAX_USE_DEPTH + 1)).expect_err("one more is refused");
    assert!(e.contains(&format!("nested more than {} deep", viv::MAX_USE_DEPTH)), "{e}");
}

/// `#! if-used <module>` ... `#! end-if` keeps its lines only when the
/// program imports that module, wherever in its import closure: a lookup can
/// serve whichever modules a program loads without loading them all (as
/// lib/basis/named does for the basis tables).
#[test]
fn an_if_used_block_keeps_its_lines_only_when_the_program_uses_the_module() {
    let modules = |name: &str| -> Result<String, String> {
        match name {
            "lib/x" => Ok("def x() {\n    return 1.0\n}\n".to_string()),
            "lib/y" => Ok("def y() {\n    return 2.0\n}\n".to_string()),
            "lib/pick" => Ok("def pick(k) {\n#! if-used lib/x\n    if k == 1 {\n        return x()\n    }\n#! end-if\n#! if-used lib/y\n    if k == 2 {\n        return y()\n    }\n#! end-if\n    stop(\"not loaded\")\n}\n".to_string()),
            "lib/bad" => Ok("#! if-used lib/x\nfrom y import y\n#! end-if\n".to_string()),
            "lib/open" => Ok("#! if-used lib/x\nshow 1\n".to_string()),
            other => Err(format!("no module {other}")),
        }
    };
    let mut resolve = modules;
    // lib/x is used after the lookup, and still counts
    let out = viv::expand_uses("from pick import pick\nimport x\nshow pick(1)\n", &mut resolve).expect("expands");
    assert!(out.contains("return x()") && !out.contains("return y()"), "{out}");
    assert!(!out.contains("#! if-used") && !out.contains("#! end-if"), "{out}");
    let ran = viv::interp::run(&out).expect("runs");
    assert!(ran.starts_with('1'), "{ran}");
    assert!(viv::compile_c(&out).is_ok(), "and it compiles:\n{out}");
    // with neither table, the lookup has no branch and stops
    let none = viv::expand_uses("from pick import pick\nshow pick(1)\n", &mut resolve).expect("expands");
    assert!(!none.contains("return x()") && !none.contains("return y()"), "{none}");
    assert!(viv::interp::run(&none).unwrap_err().contains("not loaded"));
    // a block chooses lines, not modules; and it must be closed
    assert!(viv::expand_uses("import bad\n", &mut resolve).unwrap_err().contains("inside an `#! if-used` block"));
    assert!(viv::expand_uses("import open\n", &mut resolve).unwrap_err().contains("never closed"));
}

/// `#! if-none-used <modules>` ... `#! end-if` keeps its lines, imports
/// included, when the program's other imports reach none of the
/// modules: a lookup that loads every table when the program names none, and
/// only the program's own when it names one (lib/basis/named).
#[test]
fn an_if_none_used_block_is_a_default_the_program_can_override() {
    let pick = "#! if-none-used x y\nfrom x import x\nfrom y import y\n#! end-if\ndef pick(k) {\n#! if-used lib/x\n    if k == 1 {\n        return x()\n    }\n#! end-if\n#! if-used lib/y\n    if k == 2 {\n        return y()\n    }\n#! end-if\n    stop(\"not loaded\")\n}\n";
    let modules = move |name: &str| -> Result<String, String> {
        match name {
            "lib/x" => Ok("def x() {\n    return 1.0\n}\n".to_string()),
            "lib/y" => Ok("def y() {\n    return 2.0\n}\n".to_string()),
            "lib/pick" => Ok(pick.to_string()),
            other => Err(format!("no module {other}")),
        }
    };
    let mut resolve = modules;
    // nothing named: both, by default
    let all = viv::expand_uses("from pick import pick\nshow pick(1) + pick(2)\n", &mut resolve).expect("expands");
    assert!(all.contains("def x()") && all.contains("def y()") && all.contains("return y()"), "{all}");
    assert!(viv::interp::run(&all).expect("runs").starts_with('3'));
    // y named, before or after the lookup: y alone, and x's branch is gone
    for src in ["import y\nfrom pick import pick\nshow pick(2)\n", "from pick import pick\nimport y\nshow pick(2)\n"] {
        let one = viv::expand_uses(src, &mut resolve).expect("expands");
        assert!(one.contains("def y()") && !one.contains("def x()") && !one.contains("return x()"), "{one}");
        assert!(viv::interp::run(&one).expect("runs").starts_with('2'));
        assert!(viv::compile_c(&one).is_ok(), "{one}");
    }
}

/// A module named twice is included once. `lib/lqr` uses `lib/are`, so a
/// program that names both used to get every definition in `lib/are` twice,
/// which the C backend rejects and the interpreter quietly tolerated.
#[test]
fn a_module_named_twice_is_included_once() {
    let modules = |name: &str| -> Result<String, String> {
        match name {
            "leaf" => Ok("def twice(x) {\n    return 2.0 * x\n}\n".to_string()),
            "middle" => Ok("from leaf import twice\ndef four(x) {\n    return twice(twice(x))\n}\n".to_string()),
            other => Err(format!("no module {other}")),
        }
    };
    let src = "from middle import four\nimport leaf\nshow four(3.0)\n";
    let mut resolve = modules;
    let out = viv::expand_uses(src, &mut resolve).expect("expands");
    assert_eq!(
        out.matches("def twice").count(),
        1,
        "leaf should appear once, not once per mention:\n{out}"
    );
    assert!(out.contains("(already included)"), "the second mention should say so:\n{out}");
    let ran = viv::interp::run(&out).expect("runs");
    assert!(ran.contains("12"), "four(3) is 12, got:\n{ran}");
    assert!(viv::compile_c(&out).is_ok(), "and it must compile:\n{out}");
}

/// A used module's tests stay in the module. A reader who presses Run on a post
/// is asking to run the post; before this, a post using `lib/chain` spent seven
/// seconds re-verifying a triple pendulum's energy drift first.
#[test]
fn a_used_modules_tests_do_not_come_with_it() {
    let modules = |name: &str| -> Result<String, String> {
        match name {
            "kit" => Ok(concat!(
                "def twice(x) {\n    return 2.0 * x\n}\n",
                "test \"twice doubles\" {\n    assert_near(twice(2.0), 4.0, 1.0e-12)\n}\n",
                "def thrice(x) {\n    return 3.0 * x\n}\n",
            )
            .to_string()),
            other => Err(format!("no module {other}")),
        }
    };
    let mut resolve = modules;
    let out = viv::expand_uses("from kit import thrice, twice\nshow thrice(twice(1.0))\n", &mut resolve).expect("expands");
    assert!(!out.contains("twice doubles"), "the module's test came along:\n{out}");
    assert!(out.contains("def twice"), "the module's code should still be here:\n{out}");
    assert!(out.contains("def thrice"), "code after the test should survive:\n{out}");
    let ran = viv::interp::run(&out).expect("runs");
    assert!(ran.contains('6'), "thrice(twice(1)) is 6, got:\n{ran}");
    assert!(!ran.contains("PASS"), "no test should have run:\n{ran}");
    // and running the module itself still runs its test
    let own = viv::interp::run(&modules("kit").unwrap()).expect("runs");
    assert!(own.contains("PASS twice doubles"), "the module must still test itself:\n{own}");
}

/// Text, the one way a Viv program says something that is not a number.
/// `chars("...")` is a string literal turned into an array of Unicode scalar
/// values at parse time, so everything downstream sees an ordinary array; and
/// `text(a)` writes one back out. Both engines must agree byte for byte,
/// including the UTF-8 encoding and the escapes.
#[test]
fn text_goes_in_as_an_array_and_comes_back_out() {
    let src = "a = chars(\"hi\")\nshow len(a)\nshow a[0]\ntext(a)\ntext(chars(\"x\\ny\"))\ntext(chars(\"π\"))\n";
    let ran = viv::interp::run(src).expect("runs");
    assert!(ran.contains("2\n"), "two characters:\n{ran}");
    assert!(ran.contains("104"), "`h` is 104:\n{ran}");
    assert!(ran.contains("text: hi"), "and comes back:\n{ran}");
    assert!(ran.contains("text: x\\ny"), "a newline is escaped so a record is a line:\n{ran}");
    assert!(ran.contains("text: π"), "and non-ASCII survives:\n{ran}");
    assert!(viv::compile_c(src).is_ok(), "and it compiles");
}

/// `chars` wants a literal rather than something computed. (A string is a
/// value; `chars` stays the door from text to an
/// array of code points.)
#[test]
fn chars_takes_a_literal() {
    assert!(viv::interp::run("a = chars(1.0)\n").is_err(), "`chars` wants a literal");
    assert!(viv::interp::run("a = chars(\"\")\n").is_err(), "an empty array is not allowed");
    assert!(viv::interp::run("text(3.0)\n").is_err(), "`text` wants an array");
}

#[test]
fn deriv_refuses_a_call_to_the_programs_own_def_of_a_builtin_name() {
    // the program's pow is a product; deriv would have differentiated x^2
    let src = "def pow(x, y) {\n    return x * y\n}\nd = deriv(fn(x) -> pow(x, 2.0), 3.0)\nshow d\n";
    let want = "`deriv` cannot differentiate through `pow`: this program defines its own `pow`";
    for (engine, r) in [("C", viv::compile_c(src).map(|_| ())), ("wasm", viv::compile_wasm(src).map(|_| ()))] {
        let e = r.expect_err(engine);
        assert!(e.contains(want), "{engine}: {e}");
    }
    let e = viv::interp::run(src).expect_err("interpreter");
    assert!(e.contains(want), "interpreter: {e}");
    // without the def, the builtin's derivative: d/dx x^2 at 3 is 6
    let out = viv::interp::run("d = deriv(fn(x) -> pow(x, 2.0), 3.0)\nshow d\n").unwrap();
    assert!(out.contains("d: 6"), "{out}");
}
