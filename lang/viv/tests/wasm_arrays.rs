// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The direct WebAssembly backend (Viv -> wasm bytes directly, no emcc) must
// handle the array layer: zeros, 1-D/2-D indexing (read + write), array
// literals, and arrays passed to functions. These check that such programs
// compile; tests/wasm_direct.rs runs them and compares what they print.

fn emit(src: &str) -> Vec<u8> {
    viv::compile_wasm(src).expect("the direct wasm backend must handle this program")
}

#[test]
fn wasm_backend_compiles_arrays() {
    let bytes = emit(
        "a = zeros(3)\n\
         a[0] = 1.0\n\
         a[1] = 2.0\n\
         s = a[0] + a[1]\n\
         show s\n\
         M = zeros(2, 2)\n\
         M[1, 1] = 5.0\n\
         show M[1, 1]\n\
         v = [7.0, 8.0]\n\
         show v[1]\n",
    );
    assert_eq!(&bytes[0..4], &[0x00, 0x61, 0x73, 0x6d], "wasm magic header");
    // arrays live in the runtime's memory, which the module imports
    assert!(bytes.windows(6).any(|w| w == b"memory"), "module must import the runtime's memory");
}

#[test]
fn wasm_backend_loop_allocation_compiles() {
    // Arrays allocated inside a loop, nested loops, and a persistent
    // accumulator allocated before the loop.
    emit(
        "acc = zeros(2)\n\
         for it to 10 {\n\
             a = zeros(4)\n\
             for j to 4 { a[j] = it * j }\n\
             acc[0] = acc[0] + a[1]\n\
         }\n\
         show acc[0]\n",
    );
}

#[test]
fn wasm_backend_arrays_in_functions() {
    // arrays passed to a function and indexed inside it must compile
    emit(
        "def dot(x, y, m) {\n\
             s = 0.0\n\
             for i to m { s = s + x[i] * y[i] }\n\
             return s\n\
         }\n\
         a = zeros(4)\n\
         for i to 4 { a[i] = i }\n\
         show dot(a, a, 4)\n",
    );
}

#[test]
fn wasm_backend_nested_indexing_compiles() {
    // An array element used as an index (`v[b[i]]`, `M[b[0], b[1]]`) must
    // compile: the inner access must not reuse the outer one's scratch locals.
    emit(
        "b = zeros(3)\n\
         v = zeros(3)\n\
         M = zeros(3, 3)\n\
         show v[b[0]]\n\
         show M[b[0], b[1]]\n",
    );
}

#[test]
fn wasm_backend_len_and_mod_builtins_compile() {
    // len(A) reads dim0 from the array header; mod(a, b) is the runtime's
    // `%`, the interpreter's (wasm has no f64 remainder opcode).
    emit(
        "a = zeros(5)\n\
         n = len(a)\n\
         show n\n\
         show mod(7.0, 3.0)\n",
    );
}

#[test]
fn wasm_backend_compiles_eig_sym() {
    // `eig_sym(A, vals, vecs)` runs as the interpreter's own builtin, through
    // the runtime, writing its out-parameters back into the module's arrays.
    emit(
        "A = zeros(2, 2)\n\
         A[0, 0] = 2.0\n\
         A[1, 1] = 3.0\n\
         vals = zeros(2)\n\
         vecs = zeros(2, 2)\n\
         eig_sym(A, vals, vecs)\n\
         show vals[0]\n",
    );
}

#[test]
fn wasm_backend_compiles_def_passed_as_value() {
    // compile_wasm runs the defunctionalization pass first: a def passed as a
    // value reaches the backend as a first-order call.
    let bytes = viv::compile_wasm(
        "def triple(x) { return x * 3.0 }\n\
         def apply_twice(f, x) { return f(f(x)) }\n\
         show apply_twice(triple, 2.0)\n",
    )
    .expect("compile_wasm must handle a def passed as a value");
    assert_eq!(&bytes[0..4], &[0x00, 0x61, 0x73, 0x6d], "wasm magic header");
}
