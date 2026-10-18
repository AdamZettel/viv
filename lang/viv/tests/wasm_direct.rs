// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The direct wasm backend, run end to end: each program is compiled to a
//! module (`viv::compile_wasm`), run in node against viv.wasm's runtime
//! (tools/run-wasm.mjs), and what it prints is compared with the interpreter's.
//!
//! Four engines, one output: the module, the interpreter compiled to wasm
//! (the Run button, `run-wasm.mjs --interp`), the native interpreter, and the
//! program compiled to C and run natively all print the same bytes. They can,
//! transcendentals included, because all of them compute sin, exp, pow and the
//! rest with Viv's own libm (`viv::vmath` and its C twin `vmath.c`).
//!
//! Skips (with a note) when node or a viv.wasm with the runtime exports is
//! missing: build it with www/build-wasm.sh.

use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("the repo root")
}

/// node, and a viv.wasm that exports the runtime, or None (and the test skips).
fn setup() -> Option<PathBuf> {
    let wasm = std::env::var("VIV_WASM").map(PathBuf::from).unwrap_or_else(|_| root().join("www/public/viv/viv.wasm"));
    let bytes = std::fs::read(&wasm).ok()?;
    if !bytes.windows(8).any(|w| w == b"vrt_init") {
        eprintln!("skip: {} has no runtime exports (www/build-wasm.sh builds one)", wasm.display());
        return None;
    }
    if Command::new("node").arg("--version").output().is_err() {
        eprintln!("skip: no node");
        return None;
    }
    Some(wasm)
}

/// Run `node tools/run-wasm.mjs <args>`: (stdout, stderr, exit code).
fn node(wasm: &Path, args: &[&str]) -> (String, String, i32) {
    let out = Command::new("node")
        .arg(root().join("tools/run-wasm.mjs"))
        .args(args)
        .env("VIV_WASM", wasm)
        .output()
        .expect("node runs");
    (String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned(), out.status.code().unwrap_or(-1))
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("viv-wasm-direct-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name)
}

/// Compile and run `src` on the direct backend: (stdout, stderr, exit code).
fn direct(wasm: &Path, name: &str, src: &str) -> (String, String, i32) {
    let bytes = viv::compile_wasm(src).unwrap_or_else(|e| panic!("{name}: the direct backend refused it: {e}"));
    let m = tmp(&format!("{name}.wasm"));
    std::fs::write(&m, bytes).unwrap();
    node(wasm, &[m.to_str().unwrap()])
}

/// The interpreter in viv.wasm on `src`.
fn interp_wasm(wasm: &Path, name: &str, src: &str) -> (String, String, i32) {
    let f = tmp(&format!("{name}.viv"));
    std::fs::write(&f, src).unwrap();
    node(wasm, &["--interp", f.to_str().unwrap()])
}

/// The native interpreter's output, without the call graph lines.
fn interp_native(src: &str) -> String {
    let out = viv::interp::run(src).unwrap_or_else(|e| format!("error: {e}\n"));
    out.lines().filter(|l| !l.starts_with("__graph__") && !l.starts_with("__trace__")).map(|l| format!("{l}\n")).collect()
}

const CONTROL: &str = r#"
def f(n) {
    s = 0.0
    for i to n {
        if i == 2.0 { continue }
        if i >= 7.0 and i > 0.0 { break }
        s = s + i
    }
    return s
}
print(f(100.0))
flag = f(3.0) > 1.0
show flag
k = 0.0
while true {
    k = k + 1.0
    if k > 4.0 { break }
}
show k
def fact(n) {
    if n <= 1.0 { return 1.0 }
    return n * fact(n - 1.0)
}
show fact(10.0)
for i to 3 {
    i = 10.0
}
show i
test "loops, break and continue" {
    assert_near(f(100.0), 19.0, 0.0)
    assert_true(not (k < 5.0))
}
"#;

const ARRAYS: &str = r#"
A = [[4.0, 1.0], [1.0, 3.0]]
vals = zeros(2)
vecs = zeros(2, 2)
eig_sym(A, vals, vecs)
show vals
x = linspace(0.0, 1.0, 5)
show x
show sum(x * x)
show max(x)
B = A @ transpose(A)
show B
C = 2.0 - A / 4.0
show C
rhs = [1.0, 2.0]
sol = zeros(2)
lin_solve(A, rhs, sol)
show sol
s = [3.0, 1.0, 2.0]
sort(s)
show s
show len(A)
show numel(A)
show shape(A)
m = max(x, 0.5)
show m
v = [1.0, 2.0, 3.0]
w = -v
show w
T = zeros(2, 3, 4)
T[1, 2, 3] = 7.0
show T[1, 2, 3]
show simpson(fn(t) -> t * t, 0.0, 1.0, 10)
show trapezoid(fn(t) -> t * t, 0.0, 1.0, 10)
narrate "halfway"
figure("a title", "x", "y", 0.0, 1.0, 0.0, 1.0)
plot(x, x * x, "square")
figure("points", "x", "y")
scatter(x, x, "pts", "red")
test "arrays" {
    assert_allclose(sol, [0.09090909090909091, 0.6363636363636364], 1.0e-15)
    assert_near(vals[0] + vals[1], 7.0, 1.0e-12)
}
"#;

const COMPLEX: &str = r#"
z = complex(1.0, 2.0)
w = z * z / (z + 1.0)
show w
show re(w)
show im(w)
show conj(z)
show 3.0 - z
a = czeros(6)
for i to 6 {
    a[i] = complex(i, 1.0)
}
fft(a)
show a
ifft(a)
show a
b = [1.0 + 2.0i, 3.0]
show b
M = czeros(2, 2)
M[0, 1] = 2.0i
show M @ M
test "complex" {
    assert_near(re(z * conj(z)), 5.0, 0.0)
}
"#;

const FUNCTIONS: &str = r#"
def sq(x) {
    return x * x
}
def cube(x) {
    return x * x * x
}
fs = [sq, cube]
for i to len(fs) {
    show apply(fs[i], 3.0)
}
g = sq
show apply(g, 5.0)
def twice(f, x) {
    return f(f(x))
}
show twice(sq, 3.0)
test "function values" {
    assert_near(apply(fs[1], 2.0), 8.0, 0.0)
}
"#;

const TRANSCENDENTAL: &str = r#"
x = linspace(0.1, 3.0, 7)
show sin(x)
show exp(x)
show pow(x, 1.5)
show atan2(1.0, 3.0)
show gamma(4.5)
show erf(0.3)
show deriv(fn(t) -> exp(t) * sin(t), 0.7)
z = complex(0.5, 0.25)
show exp(z)
show sqrt(z)
show log(z)
srand(7.0)
show rand()
"#;

/// `for i to n` reads n once, before the first pass, and counts its passes
/// with a counter of its own, in every engine: a body that grows n, a bound
/// with a side effect, a body that assigns the variable, the variable after
/// the loop, and a nested loop whose bound the inner body changes.
const FOR_BOUND: &str = r#"
def count_calls(calls) -> Num {
    calls[0] += 1
    return 3
}
n = 3
p = 0
for i to n {
    n += 1
    p += 1
    if p > 10 {
        break
    }
}
show p
calls = zeros(1)
for i to count_calls(calls) {
}
show calls[0]
for j to 4 {
    j = j + 1
    print(j)
}
show j
m = 3
t = 0
for a to m {
    for b to m {
        m += 1
        t += 1
    }
    if t > 100 {
        break
    }
}
show t
x = 2.5
h = zeros(4)
parallel for i to x {
    h[i] = 1
}
show sum(h)
"#;

/// The program compiled to C and run natively, or None without a C compiler.
fn native_c(name: &str, src: &str) -> Option<String> {
    let c = viv::compile_c(src).expect("compiles to C");
    let (cf, bin) = (tmp(&format!("{name}.c")), tmp(&format!("{name}.bin")));
    std::fs::write(&cf, c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let built = Command::new(&cc).args(["-std=c99", "-O2"]).arg(&cf).arg("-lm").arg("-o").arg(&bin).output().ok()?;
    assert!(built.status.success(), "{name}: the C did not compile:\n{}", String::from_utf8_lossy(&built.stderr));
    let out = Command::new(&bin).output().unwrap();
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

#[test]
fn direct_wasm_matches_the_interpreter() {
    let Some(wasm) = setup() else { return };
    for (name, src) in [
        ("control", CONTROL),
        ("arrays", ARRAYS),
        ("complex", COMPLEX),
        ("functions", FUNCTIONS),
        ("transcendental", TRANSCENDENTAL),
        ("for_bound", FOR_BOUND),
    ] {
        let (d_out, d_err, d_code) = direct(&wasm, name, src);
        assert_eq!(d_code, 0, "{name}: the module failed: {d_err}");
        assert!(!d_out.is_empty(), "{name}: printed nothing");
        let (i_out, _, _) = interp_wasm(&wasm, name, src);
        assert_eq!(d_out, i_out, "{name}: direct wasm and the interpreter in wasm disagree");
        assert_eq!(d_out, interp_native(src), "{name}: direct wasm and the native interpreter disagree");
        // Native C too. CONTROL used to be left out: after `for i to 3 { i = 10.0 }`
        // the C build printed i: 11 where the interpreter and the module print
        // 10, because the C loop counted with the variable itself. It counts
        // with a counter of its own now, and reads the bound once (FOR_BOUND).
        if let Some(c_out) = native_c(name, src) {
            assert_eq!(d_out, c_out, "{name}: direct wasm and native C disagree");
        }
    }
}

/// Complex arithmetic gives the same bits in every engine: the probe prints
/// the IEEE bit pattern of each of ~300 quotients, products and complex
/// functions (random, and adversarial: huge, tiny, subnormal, zero, infinite
/// and NaN parts). Division is Smith's algorithm everywhere; the C engine used
/// C's own `/` on `double complex`, which differed from the interpreter's
/// textbook formula in the last bit.
#[test]
fn complex_arithmetic_is_bit_identical_in_every_engine() {
    let src = include_str!("../../../study/idioms/syntax-probes/complex_division.viv");
    let i_out = interp_native(src);
    assert!(i_out.lines().filter(|l| l.starts_with("text: ")).count() > 300, "the probe printed too little:\n{i_out}");
    assert!(!i_out.contains("FAIL") && !i_out.starts_with("error"), "the probe's tests failed in the interpreter:\n{i_out}");
    if let Some(c_out) = native_c("complex_division", src) {
        assert_eq!(c_out, i_out, "native C and the interpreter disagree");
    }
    let Some(wasm) = setup() else { return };
    let (d_out, d_err, d_code) = direct(&wasm, "complex_division", src);
    assert_eq!(d_code, 0, "the module failed: {d_err}");
    assert_eq!(d_out, i_out, "direct wasm and the interpreter disagree");
}

/// `eig_sym` orders equal eigenvalues the same way in every engine (a stable
/// sort): the C runtime's exchange sort swapped the eigenvectors of a
/// degenerate eigenvalue, so sums over them rounded differently.
#[test]
fn eig_sym_orders_ties_the_same_in_every_engine() {
    let src = include_str!("../../../study/idioms/syntax-probes/eig_sym_ties.viv");
    let i_out = interp_native(src);
    assert!(i_out.contains("PASS ") && !i_out.contains("FAIL"), "the probe failed in the interpreter:\n{i_out}");
    if let Some(c_out) = native_c("eig_sym_ties", src) {
        assert_eq!(c_out, i_out, "native C and the interpreter disagree");
    }
    let Some(wasm) = setup() else { return };
    let (d_out, d_err, d_code) = direct(&wasm, "eig_sym_ties", src);
    assert_eq!(d_code, 0, "the module failed: {d_err}");
    assert_eq!(d_out, i_out, "direct wasm and the interpreter disagree");
}

#[test]
fn direct_wasm_stops_with_the_interpreters_message() {
    let Some(wasm) = setup() else { return };
    for (name, src, msg) in [
        ("oob", "a = zeros(3)\nprint(1.0)\nshow a[3]\n", "error: index 3 out of bounds for dimension 0 (size 3)"),
        ("neg", "a = zeros(2, 2)\na[1, -3] = 1.0\n", "error: index -3 out of bounds for dimension 1 (size 2)"),
        ("rank", "def f(a) {\n    return a[0, 0]\n}\nshow f(zeros(3))\n", "error: array has rank 1 but was indexed with 2 index/indices"),
        ("shape", "a = zeros(2)\nb = zeros(3)\nc = a + b\n", "error: array shape mismatch: [2] vs [3]"),
    ] {
        let (out, err, code) = direct(&wasm, name, src);
        assert_eq!(code, 4, "{name}: expected a failed run (4, a run-time error)");
        // what the program printed before it stopped, as `viv run` keeps it
        let before = if name == "oob" { "1\n" } else { "" };
        assert_eq!(out, before, "{name}: the output before the error");
        assert_eq!(err.trim(), msg, "{name}");
        assert_eq!(interp_native(src).trim(), msg, "{name}: the interpreter's own message changed");
    }
}

#[test]
fn direct_wasm_reports_failed_tests_like_viv_run() {
    let Some(wasm) = setup() else { return };
    let src = "test \"two plus two\" {\n    assert_near(2.0 + 2.0, 5.0, 0.1)\n}\nassert_gt(1.0, 2.0)\n";
    let (out, err, code) = direct(&wasm, "fails", src);
    assert_eq!(code, 1);
    assert_eq!(out, interp_native(src));
    assert!(out.starts_with("FAIL two plus two -- assert_near: |4 - 5| = 1 > 0.1\n"), "{out}");
    assert_eq!(err.trim(), "2 assertions FAILED");
}

#[test]
fn direct_wasm_runs_records() {
    // The interpreter does not run records; the C backend does, and this is
    // what it prints for this program.
    let Some(wasm) = setup() else { return };
    let src = "record P { x: Num, y: Num, v: Arr }\n\
               def norm2(p: P) {\n    return p.x * p.x + p.y * p.y + p.v[0]\n}\n\
               def make(a) -> P {\n    return P { x: a, y: 2.0 * a, v: [a, a] }\n}\n\
               q = make(3.0)\nshow norm2(q)\nshow q.v\n";
    let (out, err, code) = direct(&wasm, "records", src);
    assert_eq!(code, 0, "{err}");
    assert_eq!(out, "48\n[3, 3]\n");
}

#[test]
fn direct_wasm_runs_the_debug_lens() {
    // A program with a debug lens: its step events and its story come out as
    // the interpreter writes them (the `__steps__` / `__story__` lines). The
    // `.debug` file is joined to the source as `viv` does it (read_program).
    let Some(wasm) = setup() else { return };
    let dir = root().join("lang/viv/examples/lenses");
    let debug = std::fs::read_to_string(dir.join("scf_demo.debug")).unwrap();
    let src = format!("{}\n{}", std::fs::read_to_string(dir.join("scf_demo.viv")).unwrap(), viv::debug_lens::file_to_source(&debug));
    let (out, err, code) = direct(&wasm, "scf_demo", &src);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("\n__story__ {\"total\":"), "no story in: {out}");
    let (i_out, _, _) = interp_wasm(&wasm, "scf_demo", &src);
    assert_eq!(out, i_out);
}

/// Every builtin name a program can call, as the interpreter and the checked
/// lowering know them: the math and array builtins, the figure and animation
/// verbs, the bit builtins, and `where`/`all`/`any`.
const BUILTIN_NAMES: &[&str] = &[
    "abs", "acos", "all", "any", "apart", "apply", "asin", "assert_allclose", "assert_gt", "assert_lt",
    "assert_near", "assert_true", "atan", "atan2", "axes", "band", "bars", "bnot", "bor", "bxor",
    "ceil", "checkpoint", "cholesky", "circle", "clim", "colormap", "complex", "conj", "cos", "cosh",
    "curve", "czeros", "deriv", "dots", "draw", "eig_general", "eig_herm", "eig_near", "eig_sym", "erf",
    "exp", "fabs", "fade_in", "fade_out", "fft", "fft3", "figure", "fill", "floor", "frame",
    "gamma", "histogram", "hline", "i32", "ifft", "ifft3", "im", "label", "len", "levels",
    "lin_solve", "lin_solve_multi", "linspace", "ln", "log", "max", "min", "mod", "morph", "move_to",
    "numel", "origin", "plot", "plot2d", "plot3d", "poly", "pow", "print", "put", "quiver",
    "rand", "re", "rect", "release_mode", "roll", "round", "sar", "say", "scatter", "scene",
    "segment", "setslab", "shape", "shl", "shr", "simpson", "sin", "sinh", "slab", "sort",
    "spectrum_sym", "sqrt", "srand", "stage", "style", "sum", "take", "tan", "tanh", "text",
    "together", "transpose", "trapezoid", "tridiag_solve", "u32", "verts", "vline", "wait", "where", "wire",
    "xscale", "xtick", "yscale", "ytick", "zeros",
];

/// A program's own def wins over a builtin of the same name, in every engine.
/// The interpreter always called the def; the compiled engines called the
/// builtin for some names (`band`, `mod`, `min`, `zeros`), refused to build
/// for others (`pow`, `atan2`: the C redefined the runtime's function), and
/// for the rest every engine refused the call for having the builtin's
/// argument count. One program defines all of them, two arguments each, and
/// calls each one.
#[test]
fn a_programs_own_def_wins_over_every_builtin_in_every_engine() {
    let mut src = String::new();
    for name in BUILTIN_NAMES {
        src.push_str(&format!("def {name}(a, b) {{\n    return a * 1000 + b\n}}\n"));
    }
    let mut want = String::new();
    for (k, name) in BUILTIN_NAMES.iter().enumerate() {
        src.push_str(&format!("show {name}({}, 7)\n", k + 1));
        want.push_str(&format!("{}\n", (k + 1) * 1000 + 7));
    }
    assert_eq!(interp_native(&src), want, "the interpreter");
    if let Some(c_out) = native_c("shadow_builtins", &src) {
        assert_eq!(c_out, want, "native C");
    }
    let Some(wasm) = setup() else { return };
    let (d_out, d_err, d_code) = direct(&wasm, "shadow_builtins", &src);
    assert_eq!(d_code, 0, "the module failed: {d_err}");
    assert_eq!(d_out, want, "direct wasm");
}
