// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! GPU kernels (crate::cuda): the emulator runs a kernel
//! thread by thread and stops on what a GPU would let pass (an index out of
//! range, shared memory read before it is written, a race between threads
//! or blocks); the front end refuses what cannot run on a GPU, with the
//! location; and the CUDA C is written for each kind of kernel. The bitwise
//! agreement of the examples with the CPU reference is checked by their own
//! tests (tools/three-engines.sh examples/cuda) and with numba's CUDA
//! simulator by python/tests/test_cuda_sim.py.

use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("the repo root")
}

/// What the interpreter prints, less the call graph and trace lines.
fn run(src: &str) -> String {
    viv::interp::run(src).expect("runs").lines().filter(|l| !l.starts_with("__")).map(|l| format!("{l}\n")).collect()
}

/// The run-time error a program stops with, in the interpreter.
fn stops(src: &str) -> String {
    viv::interp::run(src).expect_err("the emulator stops it")
}

/// The headline of the diagnostic every engine refuses the program with.
fn refused(src: &str) -> String {
    let ds = viv::build_ir(src, knotkit::span::SourceId(0)).err().expect("refused");
    let i = viv::interp::run(src).expect_err("the interpreter refuses it");
    assert!(i.contains(&ds[0].message), "{i}");
    viv::compile_wasm(src).expect_err("the wasm backend refuses it");
    ds[0].message.clone()
}

const REVERSE: &str = "kernel def rev(x: Arr, y: Arr) {
    s = shared(4)
    t = thread_id()
    s[t] = x[t]
    barrier()
    y[t] = s[3 - t]
}
schedule rev(blocks = 1, threads = 4)
show rev([1.0, 2.0, 3.0, 4.0], zeros(4))
";

#[test]
fn a_kernel_with_a_barrier_runs_and_its_threads_see_each_others_writes() {
    assert!(run(REVERSE).ends_with("[4, 3, 2, 1]\n"), "{}", run(REVERSE));
}

#[test]
fn the_same_kernel_in_c_and_wasm_compiles() {
    viv::compile_c(REVERSE).expect("compiles to C");
    viv::compile_wasm(REVERSE).expect("compiles to wasm");
}

#[test]
fn a_missing_barrier_stops_the_run() {
    let e = stops(&REVERSE.replace("    barrier()\n", ""));
    assert!(e.contains("kernel rev, block 0, thread 0: reads s[3], which no thread of the block has written yet (a barrier() missing"), "{e}");
}

#[test]
fn a_read_after_another_threads_write_in_one_phase_is_a_race() {
    let e = stops(&REVERSE.replace("    y[t] = s[3 - t]\n", "    y[t] = s[3 - t]\n    s[t] = 0.0\n"));
    assert!(e.contains("thread 2: reads s[1], which thread 1 wrote with no barrier() between them"), "{e}");
}

#[test]
fn an_index_past_the_end_stops_the_run() {
    let src = "kernel def dbl(x: Arr, y: Arr) {
    i = block_id() * block_size() + thread_id()
    y[i] = 2.0 * x[i]
}
schedule dbl(blocks = 2, threads = 4)
show dbl([1.0, 2.0, 3.0, 4.0, 5.0], zeros(5))
";
    let e = stops(src);
    assert!(e.contains("kernel dbl, block 1, thread 1: index 5 is outside `x` (length 5)"), "{e}");
}

#[test]
fn two_blocks_writing_one_element_is_a_race() {
    let e = stops("kernel def bw(y: Arr) {\n    y[thread_id()] = 1.0 * block_id()\n}\nschedule bw(blocks = 2, threads = 4)\nshow bw(zeros(4))\n");
    assert!(e.contains("block 1, thread 0: writes y[0], which block 0 also wrote in this launch"), "{e}");
}

#[test]
fn a_write_after_another_threads_read_in_one_phase_is_a_race() {
    let e = stops("kernel def shift(y: Arr) {\n    t = thread_id()\n    if t < 3 {\n        y[t] = y[t + 1]\n    }\n}\nschedule shift(blocks = 1, threads = 4)\nshow shift([1.0, 2.0, 3.0, 4.0])\n");
    assert!(e.contains("thread 1: writes y[1], which thread 0 read with no barrier() between them"), "{e}");
}

#[test]
fn a_barrier_under_a_per_thread_condition_is_refused() {
    let m = refused("kernel def nu(y: Arr) {\n    t = thread_id()\n    if t < 2 {\n        barrier()\n    }\n    y[t] = 1.0\n}\nschedule nu(blocks = 1, threads = 4)\nshow nu(zeros(4))\n");
    assert!(m.starts_with("`barrier()` under a condition that can differ between the threads"), "{m}");
}

#[test]
fn a_barrier_in_a_loop_every_thread_runs_alike_is_fine() {
    // a tree sum over 8 values: the loop's count is the same for every thread
    let src = "kernel def tree(x: Arr, y: Arr) {
    s = shared(8)
    t = thread_id()
    s[t] = x[t]
    stride: Int = 4
    while stride > 0 {
        barrier()
        if t < stride {
            s[t] = s[t] + s[t + stride]
        }
        stride = stride // 2
    }
    barrier()
    if t == 0 {
        y[0] = s[0]
    }
}
schedule tree(blocks = 1, threads = 8)
show tree([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], zeros(1))
";
    assert!(run(src).ends_with("[36]\n"), "{}", run(src));
}

#[test]
fn a_per_thread_value_read_after_a_barrier_is_refused() {
    let m = refused("kernel def cr(x: Arr, y: Arr) {\n    t = thread_id()\n    v = x[t] * 2.0\n    barrier()\n    y[t] = v\n}\nschedule cr(blocks = 1, threads = 4)\nshow cr([1.0, 2.0, 3.0, 4.0], zeros(4))\n");
    assert!(m.starts_with("`v` is a per-thread value set before a barrier"), "{m}");
}

#[test]
fn what_cannot_run_on_a_gpu_is_refused_with_its_location() {
    let kernel = |line: &str| format!("kernel def k(x: Arr, y: Arr) {{\n    t = thread_id()\n    {line}\n    y[t] = 1.0\n}}\nschedule k(blocks = 1, threads = 2)\nshow k([1.0, 2.0], zeros(2))\n");
    assert_eq!(refused(&kernel("z = append(x, 1.0)")), "`append` cannot run in a GPU kernel (it allocates)");
    assert_eq!(refused(&kernel("print(t)")), "a GPU kernel cannot hold a call made for its effect");
    assert_eq!(refused(&kernel("show t")), "a GPU kernel cannot hold `show` or `narrate` (no output from a GPU)");
    assert_eq!(refused(&kernel("z = \"text\"")), "a GPU kernel cannot use strings");
    assert_eq!(refused(&kernel("x = y")), "a kernel cannot rebind its parameter `x`");
    let calls = "def twice(v: Num) -> Num {\n    return 2.0 * v\n}\n".to_string() + &kernel("z = twice(1.0)");
    assert!(refused(&calls).contains("device functions are Phase 2"));
}

#[test]
fn without_a_schedule_the_grid_covers_the_first_array() {
    // 256 threads to a block: 600 elements take 3 blocks, and every thread
    // past the end returns at the guard
    let src = "kernel def dbl(x: Arr, y: Arr) {\n    i = block_id() * block_size() + thread_id()\n    if i < len(x) {\n        y[i] = 2.0 * x[i]\n    }\n}\ny = dbl(linspace(0.0, 1.0, 600), zeros(600))\nshow y[599]\nshow len(y)\n";
    assert!(run(src).ends_with("2\n600\n"), "{}", run(src));
}

#[test]
fn a_schedule_line_is_checked_against_its_kernel() {
    let k = "kernel def dbl(x: Arr, y: Arr) {\n    i = thread_id()\n    y[i] = 2.0 * x[i]\n}\n";
    let call = "show dbl([1.0, 2.0], zeros(2))\n";
    assert!(run(&format!("{k}schedule dbl(threads = len(x))\n{call}")).ends_with("[2, 4]\n"));
    assert!(refused(&format!("{k}schedule dbl(warps = 2)\n{call}")).contains("`warps` is not one of them"));
    assert!(refused(&format!("{k}schedule dbl(threads = 2, threads = 4)\n{call}")).contains("`threads` is given twice"));
    assert!(refused(&format!("{k}schedule dbl(threads = 2)\nschedule dbl(blocks = 1)\n{call}")).contains("a second schedule line for `dbl`"));
    assert!(refused(&format!("{k}schedule dbl(threads = x[0])\n{call}")).starts_with("a grid size is an integer expression"));
    assert!(refused(&format!("{k}schedule twice(threads = 2)\n{call}")).contains("names no kernel"));
    let ew = "kernel def axpy(a: Num, x: Arr, y: Arr) -> Arr {\n    return a * x + y\n}\nschedule axpy(threads = 64)\nshow axpy(2.0, [1.0], [1.0])\n";
    assert!(refused(ew).contains("is an elementwise kernel"));
}

#[test]
fn an_elementwise_kernel_is_one_thread_per_element() {
    let src = "kernel def axpy(a: Num, x: Arr, y: Arr) -> Arr {\n    return a * x + y\n}\nshow axpy(2.0, [1.0, 2.0, 3.0], [0.5, 0.5, 0.5])\n";
    assert!(run(src).ends_with("[2.5, 4.5, 6.5]\n"), "{}", run(src));
    let e = stops(&src.replace("[0.5, 0.5, 0.5]", "[0.5, 0.5]"));
    assert!(e.contains("elementwise kernel axpy: `y` has 2 elements and `x` has 3"), "{e}");
}

#[test]
fn the_cuda_c_of_each_kind_of_kernel() {
    let emit = |name: &str| {
        let src = std::fs::read_to_string(root().join(format!("examples/cuda/{name}.viv"))).unwrap();
        let toks = viv::lexer::lex(&src, knotkit::span::SourceId(0)).unwrap();
        let p = viv::cuda::raw(|| viv::parser::parse(toks, knotkit::span::SourceId(0))).unwrap();
        viv::cuda::emit(&p, &src, name).unwrap()
    };
    let e = emit("elementwise");
    assert!(e.contains("__global__ void viv_kernel_damped(") && e.contains("viv_exp_cr("), "Viv's correctly rounded exp, as the CPU's");
    let h = emit("heat");
    assert!(h.contains("__global__ void viv_kernel_heat_loop_pfor1(") && h.contains("if (v_i >= viv_n) return;"));
    let p = emit("pairwise");
    assert!(p.contains("__shared__ double v_lanes[1024];") && p.contains("__shared__ double v_nodes[255];"));
    assert_eq!(p.matches("__syncthreads();").count(), 3);
    assert!(p.contains("extern \"C\" int viv_launch_pairwise_blocks(const double *v_x, long long v_x_len,"));
    // the grid as the schedule line gives it, computed by the launcher
    assert!(p.contains("    long long threads = (8LL * v_leaf_start_len);\n    long long blocks = v_spans;\n"));
    // CUDA C++ has no `_Complex`: the device libm leaves the complex functions out
    assert!(!p.lines().any(|l| (l.starts_with("static") || l.starts_with("typedef")) && l.contains("_Complex")));
}

// ---------------------------------------------------------------- `on device`

const STEP: &str = "kernel def step(u: Arr, v: Arr, r: Num) {
    i = block_id() * block_size() + thread_id()
    n = len(u)
    if i < n {
        if i == 0 or i == n - 1 {
            v[i] = u[i]
        }
        else {
            v[i] = u[i] + r * (u[i - 1] - 2.0 * u[i] + u[i + 1])
        }
    }
}
kernel def scale(a: Num, x: Arr) -> Arr {
    return a * x
}
";

fn device(body: &str) -> String {
    format!("{STEP}u = [0.0, 0.0, 4.0, 0.0, 0.0]\nv = copy(u)\nw = zeros(5)\non device u, v, w {{\n{body}\n}}\nshow u\nshow device_transfers()\n")
}

#[test]
fn a_time_loop_on_the_device_copies_each_array_once_each_way() {
    let src = device("    for s to 100 {\n        v = step(u, v, 0.25)\n        u, v = v, u\n    }\n    w = scale(2.0, u)");
    let out = run(&src);
    assert!(out.ends_with("[3, 3, 101]\n"), "{out}");
    viv::compile_c(&src).expect("compiles to C");
    viv::compile_wasm(&src).expect("compiles to wasm");
}

#[test]
fn a_kernel_call_outside_a_block_counts_its_copies() {
    let src = format!("{STEP}u = [0.0, 1.0, 0.0]\nv = step(u, zeros(3), 0.25)\ny = scale(2.0, u)\nshow device_transfers()\n");
    // step: two arrays in, the written one out; scale: x and its result in, the result out
    assert!(run(&src).ends_with("[4, 2, 2]\n"), "{}", run(&src));
}

#[test]
fn the_arrays_held_before_the_block_are_not_changed_by_it() {
    let src = format!("{STEP}u = [0.0, 0.0, 4.0, 0.0, 0.0]\nalias = u\nv = copy(u)\non device u, v {{\n    v = step(u, v, 0.25)\n    u, v = v, u\n}}\nshow alias\nshow u\n");
    assert!(run(&src).ends_with("alias: [0, 0, 4, 0, 0]\nu: [0, 1, 2, 1, 0]\n"), "{}", run(&src));
}

#[test]
fn the_host_cannot_touch_a_device_array_inside_the_block() {
    for (body, says) in [
        ("    show u", "`u` is on the device inside `on device`"),
        ("    x = u[2]", "`u` is on the device inside `on device`"),
        ("    u = zeros(5)", "`u` is on the device here"),
        ("    q = step(u, v, 0.25)", "the result of `step` goes to one of the block's arrays"),
        ("    u = step(u, v, 0.25)", "`step` writes its `v`, given `v` here"),
        ("    v = step(u, zeros(5), 0.25)", "is one of the block's arrays"),
        ("    v = step(v, v, 0.25)", "is passed to `step` twice"),
        ("    u, v = v, w", "swapped among themselves"),
        ("    x = sum(scale(2.0, u))", "a kernel call is a statement of its own"),
        ("    return", "would leave before the arrays come back"),
        ("    for s to 3 {\n        on device w {\n        }\n    }", "inside another"),
    ] {
        let src = device(body);
        let msg = refused(&src);
        assert!(msg.contains(says), "{body}: {msg}");
    }
    let src = format!("def peek(a: Arr) -> Num {{\n    return a[0]\n}}\n{}", device("    x = peek(w)"));
    assert!(refused(&src).contains("`peek` is a def"), "{}", refused(&src));
    let src = format!("{STEP}u = zeros(3)\nfor k to 2 {{\n    on device u {{\n        break\n    }}\n}}\n");
    assert!(refused(&src).contains("leave the `on device` block"), "{}", refused(&src));
}

#[test]
fn the_cuda_c_of_a_block_copies_only_at_its_ends() {
    let src = std::fs::read_to_string(root().join("examples/cuda/heat2d.viv")).unwrap();
    let toks = viv::lexer::lex(&src, knotkit::span::SourceId(0)).unwrap();
    let p = viv::cuda::raw(|| viv::parser::parse(toks, knotkit::span::SourceId(0))).unwrap();
    let c = viv::cuda::emit(&p, &src, "heat2d").unwrap();
    let start = c.find("extern \"C\" int viv_region_heat2d_device_1(double *h_u, long long h_u_len, double *h_v, long long h_v_len, long long v_w, double v_r, long long v_steps, long long v_h)").expect("the block's host function");
    let region = &c[start..];
    let region = &region[..region.find("\n}\n").unwrap()];
    assert_eq!(region.matches("cudaMemcpyHostToDevice").count(), 2);
    assert_eq!(region.matches("cudaMemcpyDeviceToHost").count(), 2);
    assert_eq!(region.matches("cudaDeviceSynchronize").count(), 1);
    let looped = &region[region.find("for (v_s = 0;").unwrap()..region.find("cudaDeviceSynchronize").unwrap()];
    assert!(looped.contains("VIV_CUDA_CHECK(viv_dlaunch_heat2d(v_u, v_u_len, v_v, v_v_len, v_w, v_h, v_r));") && !looped.contains("cudaMemcpy") && !looped.contains("cudaMalloc"), "{looped}");
    assert!(c.contains("extern \"C\" int viv_dlaunch_heat2d(") && !c.contains("not emitted: the `on device`"));
}

// ---------------------------------------------------------------- 2-D grids

/// Each thread writes its (row, column) and the grid's sizes along y into
/// its element; 5 x 3 elements under blocks of 2 x 2 threads leave threads
/// past the edge on both axes.
const GRID2: &str = "kernel def where(y: Arr, w: Int, h: Int) {
    col = block_id(0) * block_size(0) + thread_id(0)
    row = block_id(1) * block_size(1) + thread_id(1)
    if col < w and row < h {
        y[row * w + col] = 100 * row + col + 0.25 * block_size(1) + 0.5 * grid_size(1)
    }
}
schedule where(threads = [2, 2], cover = [w, h])
show where(zeros(15), 5, 3)
";

#[test]
fn a_2d_grid_gives_each_thread_its_row_and_column() {
    // block_size(1) = 2 and grid_size(1) = 2 (3 rows in blocks of 2): + 0.5 + 1
    let out = run(GRID2);
    assert!(out.ends_with("[1.5, 2.5, 3.5, 4.5, 5.5, 101.5, 102.5, 103.5, 104.5, 105.5, 201.5, 202.5, 203.5, 204.5, 205.5]\n"), "{out}");
    viv::compile_c(GRID2).expect("compiles to C");
    viv::compile_wasm(GRID2).expect("compiles to wasm");
}

#[test]
fn a_2d_grid_from_blocks_and_the_default_axis() {
    // blocks given along both axes; thread_id() is thread_id(0)
    let src = "kernel def k(y: Arr) {\n    i = (block_id(1) * grid_size(0) + block_id()) * block_size(0) * block_size(1) + thread_id(1) * block_size() + thread_id()\n    y[i] = i\n}\nschedule k(blocks = [2, 3], threads = [4, 2])\nshow sum(k(zeros(48)))\n";
    assert!(run(src).ends_with("1128\n"), "{}", run(src));
}

#[test]
fn a_race_between_rows_of_a_2d_block_stops_the_run() {
    // the threads of row 0 and row 1 of a block write one element each column
    let src = "kernel def k(y: Arr) {\n    y[thread_id(0)] = thread_id(1)\n}\nschedule k(threads = [4, 2], blocks = 1)\nshow k(zeros(4))\n";
    let e = stops(src);
    assert!(e.contains("also wrote with no barrier() between them"), "{e}");
}

#[test]
fn a_2d_schedule_is_checked() {
    let k = "kernel def k(y: Arr) {\n    y[thread_id(0)] = 1.0\n}\n";
    let call = "show k(zeros(4))\n";
    assert!(refused(&format!("{k}schedule k(threads = [4, 1, 1])\n{call}")).contains("one value per axis"));
    assert!(refused(&format!("{k}schedule k(blocks = [1, 1], cover = [4, 1])\n{call}")).contains("gives both `blocks` and `cover`"));
    assert!(refused(&format!("{k}schedule k(cover = [y[0], 1])\n{call}")).starts_with("a grid size is an integer expression"));
    let bad_axis = "kernel def k(y: Arr) {\n    y[thread_id(2)] = 1.0\n}\nshow k(zeros(4))\n";
    assert!(refused(bad_axis).contains("an axis written as a number"));
}

fn emit_example(name: &str, sets: &[(&str, &str)]) -> String {
    let mut src = std::fs::read_to_string(root().join(format!("examples/cuda/{name}.viv"))).unwrap();
    for (k, v) in sets {
        let at = src.find(&format!("\n{k}: Int = ")).expect("the constant") + 1;
        let end = at + src[at..].find('\n').unwrap();
        src.replace_range(at..end, &format!("{k}: Int = {v}"));
    }
    let toks = viv::lexer::lex(&src, knotkit::span::SourceId(0)).unwrap();
    let p = viv::cuda::raw(|| viv::parser::parse(toks, knotkit::span::SourceId(0))).unwrap();
    viv::cuda::emit(&p, &src, name).unwrap()
}

fn kernel_text<'a>(c: &'a str, name: &str) -> &'a str {
    let start = c.find(&format!("__global__ void viv_kernel_{name}(")).expect("the kernel");
    &c[start..start + c[start..].find("\n}\n").unwrap()]
}

#[test]
fn the_cuda_c_of_a_2d_grid() {
    let c = emit_example("heat2d", &[("TX", "64"), ("TY", "4")]);
    // the 2-D kernel reads its row and column from y and x, with no division
    let k2 = kernel_text(&c, "heat2d");
    assert!(k2.contains("(long long)blockIdx.y * (long long)blockDim.y) + (long long)threadIdx.y") && !k2.contains("viv_floordiv") && !k2.contains("viv_floormod"), "{k2}");
    assert!(kernel_text(&c, "heat2d_flat").contains("viv_floordiv_ll(v_i, v_w)"));
    // the grid from `cover = [w, h]` and the constants set
    assert!(c.contains("    long long threads = 64LL;\n    long long threads_y = 4LL;\n    long long blocks = (v_w + threads - 1) / threads;\n    long long blocks_y = (v_h + threads_y - 1) / threads_y;\n"), "{c}");
    assert!(c.contains("viv_kernel_heat2d<<<dim3((unsigned)blocks, (unsigned)blocks_y), dim3((unsigned)threads, (unsigned)threads_y)>>>("));
    // a 1-D kernel keeps its 1-D launch
    assert!(c.contains("viv_kernel_heat2d_flat<<<(unsigned)blocks, (unsigned)threads>>>("));
}

#[test]
fn the_cuda_c_of_the_matmul_kernels() {
    let c = emit_example("matmul", &[("TILE", "32")]);
    let t = kernel_text(&c, "matmul_tiled");
    assert_eq!(t.matches("__shared__ double").count(), 3);
    assert!(t.contains("__shared__ double v_sa[1024];"), "{t}");
    assert_eq!(t.matches("__syncthreads();").count(), 2);
    // no fused multiply-add is written; the build flags forbid nvcc's (-fmad=false)
    assert!(t.contains("v_acc = (v_acc + (v_x * v_sb[(long long)(((v_p * 32LL) + v_tx))]));"), "{t}");
    let n = kernel_text(&c, "matmul_naive");
    assert!(n.contains("v_acc = (v_acc + (v_x * v_b[(long long)(((v_p * v_n) + v_col))]));"), "{n}");
    assert!(c.contains("extern \"C\" int viv_dlaunch_matmul_tiled(const double *v_a, long long v_a_len, const double *v_b, long long v_b_len, double *v_c, long long v_c_len, long long v_m, long long v_k, long long v_n)"));
    assert!(c.contains("    long long threads = 32LL;\n    long long threads_y = 32LL;\n    long long blocks = (v_n + threads - 1) / threads;\n    long long blocks_y = (v_m + threads_y - 1) / threads_y;\n"));
}
