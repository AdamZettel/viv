// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Records that hold arrays and records, arrays of records, and writes into
//! fields: `model.blocks[l].attn.Wq = W`.
//! Every program runs in the interpreter, as C, and (when node and a viv.wasm
//! with the runtime are present) as a module from the direct wasm backend,
//! and the three must print the same bytes.

use std::path::{Path, PathBuf};
use std::process::Command;

fn viv_bin() -> &'static str {
    env!("CARGO_BIN_EXE_viv")
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("the repo root")
}

fn cc() -> String {
    std::env::var("CC").unwrap_or_else(|_| "cc".to_string())
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("viv_objects_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[derive(Debug, PartialEq)]
struct Ran {
    code: Option<i32>,
    out: String,
    err: String,
}

fn ran(o: std::process::Output) -> Ran {
    Ran { code: o.status.code(), out: String::from_utf8_lossy(&o.stdout).into_owned(), err: String::from_utf8_lossy(&o.stderr).into_owned() }
}

fn interp(dir: &Path) -> Ran {
    ran(Command::new(viv_bin()).current_dir(dir).args(["run", "-q", "p.viv"]).output().unwrap())
}

fn native(dir: &Path, sanitize: bool) -> Option<Ran> {
    if Command::new(cc()).arg("--version").output().is_err() {
        return None;
    }
    let c = Command::new(viv_bin()).current_dir(dir).args(["p.viv", "-o", "p.c"]).output().unwrap();
    assert!(c.status.success(), "viv refused the program:\n{}", String::from_utf8_lossy(&c.stderr));
    let mut args = vec!["-std=c99", "-O1", "-Wall", "-Wno-unused-function", "-Wno-unused-but-set-variable"];
    if sanitize {
        args.extend(["-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=undefined", "-fno-omit-frame-pointer"]);
    }
    let b = Command::new(cc()).current_dir(dir).args(&args).args(["p.c", "-lm", "-o", "p"]).output().unwrap();
    assert!(b.status.success(), "the C did not build:\n{}", String::from_utf8_lossy(&b.stderr));
    Some(ran(Command::new(dir.join("p")).env("ASAN_OPTIONS", "detect_leaks=0").output().unwrap()))
}

/// The direct wasm backend in node, or None when there is no runtime to run it.
fn direct(dir: &Path) -> Option<Ran> {
    let wasm = std::env::var("VIV_WASM").map(PathBuf::from).unwrap_or_else(|_| root().join("www/public/viv/viv.wasm"));
    let bytes = std::fs::read(&wasm).ok()?;
    if !bytes.windows(13).any(|w| w == b"vrt_recs_push") {
        eprintln!("skip wasm: {} has no arrays of records (www/build-wasm.sh builds one)", wasm.display());
        return None;
    }
    Command::new("node").arg("--version").output().ok()?;
    let c = Command::new(viv_bin()).current_dir(dir).args(["p.viv", "-o", "p.wasm"]).output().unwrap();
    assert!(c.status.success(), "the wasm backend refused the program:\n{}", String::from_utf8_lossy(&c.stderr));
    Some(ran(Command::new("node").arg(root().join("tools/run-wasm.mjs")).arg(dir.join("p.wasm")).env("VIV_WASM", &wasm).output().unwrap()))
}

/// Run `src` in every engine (the C under ASan and UBSan); they must agree.
/// Returns the interpreter's run.
fn all(tag: &str, src: &str) -> Ran {
    let dir = scratch(tag);
    std::fs::write(dir.join("p.viv"), src).unwrap();
    let i = interp(&dir);
    if let Some(n) = native(&dir, true) {
        assert!(!n.err.contains("Sanitizer") && !n.err.contains("runtime error"), "{tag}: a sanitizer report:\n{}", n.err);
        if i.code == Some(0) {
            assert_eq!(i, n, "interpreter and C disagree on\n{src}");
        } else {
            assert_eq!(n.code, i.code, "{src}");
            assert_eq!(n.err.lines().next(), i.err.lines().next(), "interpreter and C stop differently on\n{src}");
        }
    }
    if let Some(w) = direct(&dir) {
        if i.code == Some(0) {
            assert_eq!(i, w, "interpreter and wasm disagree on\n{src}");
        } else {
            assert_eq!(w.code, i.code, "{src}");
            assert_eq!(w.err.lines().next(), i.err.lines().next(), "interpreter and wasm stop differently on\n{src}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    i
}

fn prints(tag: &str, src: &str, out: &str) {
    let r = all(tag, src);
    assert_eq!(r.code, Some(0), "{tag}: {r:?}");
    assert_eq!(r.out, out, "{tag}");
}

/// The messages the compile gives for `src` (empty when it lowers).
fn errors(src: &str) -> Vec<String> {
    match viv::build_ir(src, knotkit::span::SourceId(0)) {
        Ok(_) => Vec::new(),
        Err(ds) => ds.into_iter().map(|d| d.message).collect(),
    }
}

const MODEL: &str = "record Attention { Wq: Arr, Wk: Arr }
record Mlp { up: Arr, down: Arr }
record Block { attn: Attention, mlp: Mlp }
record Gpt { width: Num, blocks: [Block], unembed: Arr }

def block_init(d) {
    a = Attention { Wq: zeros(d, d) + 1.0, Wk: zeros(d, d) + 2.0 }
    return Block { attn: a, mlp: Mlp { up: zeros(d, 2 * d), down: zeros(2 * d, d) } }
}

def make(d, layers) {
    blocks = []
    for l to layers {
        blocks = append(blocks, block_init(d))
    }
    return Gpt { width: d, blocks: blocks, unembed: zeros(d, 3) }
}
";

#[test]
fn a_model_is_records_of_records_and_an_array_of_them() {
    let src = format!(
        "{MODEL}
m = make(2, 3)
print(len(m.blocks), m.width, m.blocks[1].attn.Wq[0, 0], m.blocks[-1].attn.Wk[1, 1])
m.blocks[1].attn.Wq = zeros(2, 2) + 7.0
print(m.blocks[1].attn.Wq[0, 0], m.blocks[0].attn.Wq[0, 0])
m.blocks[0].mlp.down[:] = 3.0
print(m.blocks[0].mlp.down[1, 1])
m.width = 5
m.blocks[2] = block_init(2)
m.blocks[2].mlp.up[0, 0] += 4.0
print(m.width, m.blocks[2].mlp.up[0, 0])
for s to 4 {{
    m.blocks[1].attn.Wq = m.blocks[1].attn.Wq * 2.0
}}
print(m.blocks[1].attn.Wq[0, 1])
"
    );
    prints("model", &src, "3 2 1 2\n7 1\n3\n5 4\n112\n");
}

#[test]
fn records_are_values_and_their_arrays_are_shared_handles() {
    // A record, and an array of records, is a value: `b = a` gives `b` its
    // own, and a field write through one name is not seen through the other.
    // A numeric array inside is still a handle, as
    // every array is until ownership's stage 3: an element write is seen
    // through every record that holds it.
    let src = format!(
        "{MODEL}
m = make(2, 3)
b = m.blocks[2]
b.attn.Wk = zeros(2, 2) + 9.0
print(m.blocks[2].attn.Wk[0, 0], b.attn.Wk[0, 0])
m2 = m
m2.blocks[0].mlp.up = zeros(2, 4) + 5.0
m2.width = 8
print(m.blocks[0].mlp.up[0, 0], m2.blocks[0].mlp.up[0, 0], m.width, m2.width)
m2.blocks[1].attn.Wq[0, 0] = 6.0
print(m.blocks[1].attn.Wq[0, 0])
bs = m.blocks
bs[0] = block_init(4)
print(len(bs[0].attn.Wq), len(m.blocks[0].attn.Wq))
"
    );
    prints("values", &src, "2 9\n0 5 2 8\n6\n4 2\n");
}

#[test]
fn a_def_changes_its_own_copy_and_returns_it() {
    let src = format!(
        "{MODEL}
def trained(t: Gpt, steps) {{
    mine = t
    for s to steps {{
        for l to len(mine.blocks) {{
            mine.blocks[l].mlp.up = mine.blocks[l].mlp.up + 1.0
        }}
        mine.width = mine.width + 1
    }}
    return mine
}}
m = make(2, 2)
t = trained(m, 5)
print(m.width, t.width, m.blocks[1].mlp.up[0, 0], t.blocks[1].mlp.up[0, 0])
"
    );
    prints("def_copy", &src, "2 7 0 5\n");
}

#[test]
fn nested_arrays_of_records_strings_and_unpacking() {
    let src = "record Head { name: Str, w: Arr }
record Layer { heads: [Head], scale: Num }
record Net { layers: [Layer], tag: Str }

def layer(k, n) {
    hs = [Head { name: \"a\", w: zeros(n) }, Head { name: \"b\", w: zeros(n) + 1.0 }]
    if k > 0 {
        hs = append(hs, Head { name: \"c\", w: zeros(n) + k })
    }
    return Layer { heads: hs, scale: 1.0 + k }
}

def pair(k) {
    return k, Head { name: \"p\", w: zeros(2) + k }
}

net = Net { layers: [layer(0, 3), layer(2, 3)], tag: \"n\" }
print(len(net.layers[0].heads), len(net.layers[1].heads), net.layers[1].heads[2].name, sum(net.layers[1].heads[2].w))
net.layers[1].heads[0].name = \"z\"
net.tag = \"m\"
print(net.layers[1].heads[0].name, net.tag)
k, net.layers[0].heads[1] = pair(4)
print(k, net.layers[0].heads[1].name, sum(net.layers[0].heads[1].w))
";
    prints("nested", src, "2 3 c 6\nz m\n4 p 8\n");
}

#[test]
fn an_index_out_of_range_stops_every_engine_the_same_way() {
    let src = format!("{MODEL}\nm = make(2, 3)\nprint(1)\nm.blocks[3].width = 1\n");
    // a Block has no `width`: refused before it runs
    assert!(errors(&src).iter().any(|e| e.contains("has no field `width`")), "{:?}", errors(&src));
    let src = format!("{MODEL}\nm = make(2, 3)\nprint(1)\nprint(m.blocks[3].attn.Wq[0, 0])\n");
    let r = all("oob", &src);
    assert_eq!(r.code, Some(4));
    assert_eq!(r.out, "1\n");
    assert!(r.err.contains("index 3 out of bounds for dimension 0 (size 3)"), "{}", r.err);
    let src = format!("{MODEL}\nm = make(2, 3)\nm.blocks[-4] = block_init(2)\n");
    let r = all("oob_write", &src);
    assert!(r.err.contains("index -4 out of bounds for dimension 0 (size 3)"), "{}", r.err);
}

#[test]
fn memory_stays_flat_when_a_loop_writes_into_a_model() {
    // a pass stores arrays it made into a record made before the loop: the
    // loop keeps them through the model's variable, and frees what the
    // previous pass stored (study/memcheck/adversarial/record_field_writes.viv
    // measures the arena at two lengths)
    let src = format!(
        "{MODEL}
m = make(64, 3)
for k to 300 {{
    for l to len(m.blocks) {{
        m.blocks[l].attn.Wq = m.blocks[l].attn.Wq * 1.0
    }}
    m.blocks[0] = block_init(64)
}}
print(sum(m.blocks[1].attn.Wq))
"
    );
    prints("flat", &src, "4096\n");
}

#[test]
fn writes_that_would_reach_another_name_are_refused() {
    let cases: &[(&str, &str)] = &[
        // a parameter is the caller's value
        ("record V { x: Num }\ndef f(p: V) {\n    p.x = 2\n    return p.x\n}\nprint(f(V { x: 1 }))", "`p` is a parameter, and a def does not change the values it is given"),
        ("record V { x: Num }\nrecord L { vs: [V] }\ndef f(p: L) {\n    p.vs[0] = V { x: 3 }\n    return 0\n}\nprint(f(L { vs: [V { x: 1 }] }))", "`p` is a parameter, and a def does not change the values it is given"),
        // a global is the program's
        ("record V { x: Num }\ng = V { x: 1 }\ndef f() {\n    g.x = 2\n    return 0\n}\nprint(f())", "`g` is a top-level variable, and a def does not change those"),
    ];
    for (src, want) in cases {
        let errs = errors(src);
        assert!(errs.iter().any(|e| e.contains(want)), "program:\n{src}\nexpected {want:?}, got {errs:?}");
        let err = viv::interp::run(src).expect_err(src);
        assert!(err.contains(want), "the interpreter, on\n{src}\nexpected {want:?}, got {err:?}");
    }
    let refused: &[(&str, &str)] = &[
        ("record V { x: Num }\na = V { x: 1 }\na.x = zeros(2)", "the field `x` of `V` is a number but this value is an array"),
        ("record V { x: Num }\nrecord W { x: Num }\nxs = [V { x: 1 }]\nxs[0] = W { x: 2 }", "an element of this array is a V but this value is a W"),
        ("record V { x: Num }\nxs = [V { x: 1 }, 2.0]", "an array of records holds records of one type"),
        ("record V { x: Num }\nxs = [V { x: 1 }]\nshow xs", "cannot `show` a array of records"),
        ("record V { x: Num }\nxs = [V { x: 1 }]\nys = xs < xs", "records take + - * / only"),
        ("record V { x: Num }\nxs = [V { x: 1 }]\nys = copy(xs)", "`copy` takes an array of numbers or a map"),
        ("record V { x: Num }\nxs = [V { x: 1 }]\nprint(xs[0, 0].x)", "an array of records is a list, so it takes one index"),
    ];
    for (src, want) in refused {
        let errs = errors(src);
        assert!(errs.iter().any(|e| e.contains(want)), "program:\n{src}\nexpected {want:?}, got {errs:?}");
    }
}

const NET: &str = "record Lin { w: Arr, b: Arr }
record Net { layers: [Lin], head: Arr }

def lin(n, s) {
    return Lin { w: linspace(s, s + 1.0, n * n), b: zeros(n) + s }
}

def net(n, depth) {
    ls = []
    for k to depth {
        ls = append(ls, lin(n, k))
    }
    return Net { layers: ls, head: zeros(n) + 0.5 }
}
";

#[test]
fn arithmetic_over_a_records_arrays() {
    // + - * / between two records of arrays of one type, or a record and a
    // number, is the operation on each array; -r, sqrt(r), zeros_like(r)
    // and sum(r) walk the record the same way
    let src = format!(
        "{NET}
a = net(2, 3)
b = net(2, 3)
c = a + b
print(c.layers[2].b[0], c.head[1])
d = 2.0 * a - b / 4.0
print(d.layers[1].w[3], sum(d.head))
z = zeros_like(a)
print(sum(z), len(z.layers))
print(sum(a), sum(a * a))
s = sqrt(a * a + 1.0)
print(s.layers[2].w[0])
n = -a
print(n.layers[2].b[0])
"
    );
    prints("arith", &src, "4 1\n3.5 1.75\n0 3\n25 47.1667\n2.23607\n-2\n");
}

#[test]
fn for_leaves_calls_a_def_on_every_array_of_several_records() {
    // AdamW-style: the def writes its arrays in place, and the model, its
    // first and second moments are updated together, leaf by leaf
    let src = format!(
        "{NET}
def adam_leaf(w: Arr, g: Arr, m: Arr, v: Arr, t, lr) {{
    m[:] = 0.9 * m + 0.1 * g
    v[:] = 0.99 * v + 0.01 * g * g
    mhat = m / (1.0 - pow(0.9, t))
    vhat = v / (1.0 - pow(0.99, t))
    w[:] = w - lr * mhat / (sqrt(vhat) + 1.0e-8)
}}

def count(w: Arr, n: Arr) {{
    n[0] = n[0] + len(w)
}}

a = net(2, 3)
m = zeros_like(a)
v = zeros_like(a)
g = a * 0.01
for t to 3 {{
    for_leaves(adam_leaf, a, g, m, v, t + 1, 0.1)
}}
print(a.layers[1].w[2], m.head[0], v.layers[0].b[1])
n = [0.0]
for_leaves(count, a.layers, n)
print(n[0])
"
    );
    prints("leaves", &src, "1.36667 0.001355 0\n18\n");
}

#[test]
fn records_of_different_structure_do_not_combine() {
    // a record of arrays only: a number field is refused, named
    let src = "record P { x: Num, w: Arr }\na = P { x: 1, w: zeros(2) }\nb = a + a";
    assert!(errors(src).iter().any(|e| e.contains("`+` over a record acts on its arrays, and the field `x` of `P` is a number")), "{:?}", errors(src));
    let err = viv::interp::run(src).expect_err(src);
    assert!(err.contains("`+` over a record acts on its arrays, and the field `x` of `P` is a number"), "{err}");
    let src = format!("{NET}\na = net(2, 3)\nb = net(2, 2)\nc = a * b\n");
    let r = all("lengths", &src);
    assert!(r.err.starts_with("error: `*` of two arrays of Lin records needs them of one length"), "{}", r.err);
    let refused: &[(&str, &str)] = &[
        ("record V { x: Arr }\na = V { x: zeros(2) }\nb = a @ a", "records take + - * / only"),
        ("record V { x: Arr }\nrecord W { x: Arr }\na = V { x: zeros(2) }\nb = a + W { x: zeros(2) }", "`+` takes records of one type, or a record and a number"),
        ("record V { x: Arr, m: Map }\na = V { x: zeros(2), m: map() }\nb = a + a", "the field `m` of `V` is a map"),
        ("record V { x: Arr, s: Str }\na = V { x: zeros(2), s: \"a\" }\nb = sum(a)", "`sum` over a record acts on its arrays, and the field `s` of `V` is a string"),
        ("record V { x: CArr }\na = V { x: czeros(2) }\nb = zeros_like(a)", "`zeros_like` over a record acts on its arrays, and the field `x` of `V` is a complex array"),
        ("record V { x: Arr }\ndef f(w: Arr) {\n    w[0] = 1\n}\na = V { x: zeros(2) }\nfor_leaves(f, a, a)", "`f` takes 1 argument(s), and `for_leaves` passes it 2"),
        ("record V { x: Arr }\na = V { x: zeros(2) }\nfor_leaves(g, a)", "`for_leaves` calls a def by its name"),
        ("record V { x: Arr }\ndef f(w: Arr) {\n    w[0] = 1\n}\na = V { x: zeros(2) }\nparallel for i to 2 {\n    for_leaves(f, a)\n}", "a pass of a `parallel for` cannot call `for_leaves`"),
    ];
    for (src, want) in refused {
        let errs = errors(src);
        assert!(errs.iter().any(|e| e.contains(want)), "program:\n{src}\nexpected {want:?}, got {errs:?}");
    }
}

#[test]
fn value_and_grad_of_a_record_of_arrays_is_a_record_of_its_shape() {
    // loss, g = value_and_grad(loss, model, x): g is a Model, each array the
    // derivative at the same place; every one checked against central
    // differences, in every engine
    let src = format!("record Mlp {{ up: Arr, down: Arr }}
record Block {{ mlp: Mlp, gain: Arr }}
record Model {{ embed: Arr, blocks: [Block], head: Arr }}

def block_init(d, h, s) {{
    return Block {{ mlp: Mlp {{ up: s * linspace(-1.0, 1.0, d * h) , down: s * linspace(1.0, -0.5, h * d) }}, gain: zeros(d) + 1.0 + s }}
}}

def mlp(m: Mlp, X: Arr) {{
    d = len(X)
    h = len(m.up) / d
    U = zeros(h)
    for j to h {{
        for i to d {{
            U[j] = U[j] + X[i] * m.up[i * h + j]
        }}
    }}
    R = max(U, 0.0)
    Y = zeros(d)
    for i to d {{
        for j to h {{
            Y[i] = Y[i] + R[j] * R[j] * m.down[j * d + i]
        }}
    }}
    return Y
}}

def block(b: Block, X: Arr) {{
    return X + b.gain * mlp(b.mlp, X)
}}

def loss(model: Model, x: Arr) {{
    X = model.embed * x
    for l to len(model.blocks) {{
        X = block(model.blocks[l], X)
    }}
    return sum(X * model.head)
}}

m = Model {{ embed: linspace(0.5, 1.5, 3), blocks: [block_init(3, 4, 0.3), block_init(3, 4, -0.2)], head: linspace(1.0, -1.0, 3) }}
x = [1.0, 2.0, -1.0]
v, g = value_and_grad(loss, m, x)
print(v)
print(g.embed[0], g.head[2], g.blocks[0].gain[1], g.blocks[1].mlp.up[5], g.blocks[0].mlp.down[7])

def fd(m: Model, x: Arr, a: Arr, i) {{
    h = 1.0e-6
    was = a[i]
    a[i] = was + h
    up = loss(m, x)
    a[i] = was - h
    dn = loss(m, x)
    a[i] = was
    return (up - dn) / (2.0 * h)
}}
def worst_all(m: Model, g: Model, x: Arr) {{
    w = 0.0
    for i to 3 {{
        w = max(w, abs(fd(m, x, m.embed, i) - g.embed[i]))
        w = max(w, abs(fd(m, x, m.head, i) - g.head[i]))
    }}
    for l to len(m.blocks) {{
        for i to 3 {{
            w = max(w, abs(fd(m, x, m.blocks[l].gain, i) - g.blocks[l].gain[i]))
        }}
        for i to 12 {{
            w = max(w, abs(fd(m, x, m.blocks[l].mlp.down, i) - g.blocks[l].mlp.down[i]))
        }}
    }}
    return w
}}
def worst(m: Model, g: Model, x: Arr) {{
    h = 1.0e-6
    w = 0.0
    mm = m
    for l to len(m.blocks) {{
        for i to len(m.blocks[l].mlp.up) {{
            was = mm.blocks[l].mlp.up[i]
            mm.blocks[l].mlp.up[i] = was + h
            up = loss(mm, x)
            mm.blocks[l].mlp.up[i] = was - h
            dn = loss(mm, x)
            mm.blocks[l].mlp.up[i] = was
            w = max(w, abs((up - dn) / (2.0 * h) - g.blocks[l].mlp.up[i]))
        }}
    }}
    return w
}}
test \"the gradient of a record agrees with central differences\" {{
    assert_lt(worst(m, g, x), 1.0e-7)
    assert_lt(worst_all(m, g, x), 1.0e-7)
}}
");
    prints("grad_record", &src, "1.98494\n0.984652 -1.5132 0 -0.0539504 0\nPASS the gradient of a record agrees with central differences\n");
}

#[test]
fn what_grad_refuses_of_a_record() {
    let cases: &[(&str, &str)] = &[
        // a number field: grad would have to decide whether it is a size
        ("record M { n: Num, w: Arr }\ndef f(m: M) {\n    return sum(m.w)\n}\nm = M { n: 2, w: zeros(2) }\nl, g = value_and_grad(f, m)", "`grad` differentiates a record of arrays, and the field `n` of `M` is a number"),
        // arithmetic on a whole record inside the differentiated def
        ("record M { w: Arr }\ndef f(m: M) {\n    k = m * 2.0\n    return sum(k.w)\n}\nm = M { w: zeros(2) }\nl, g = value_and_grad(f, m)", "`grad` differentiates the arrays of a record, not arithmetic on whole records"),
        // a name for a part of the record, given twice
        ("record B { w: Arr }\nrecord M { bs: [B] }\ndef f(m: M) {\n    b = m.bs[0]\n    s = sum(b.w)\n    b = m.bs[1]\n    return s + sum(b.w)\n}\nm = M { bs: [B { w: zeros(2) }, B { w: zeros(2) }] }\nl, g = value_and_grad(f, m)", "names a part of the differentiated record, and is given a value more than once"),
        // a write into a record in differentiated code
        ("record M { w: Arr }\ndef f(m: M) {\n    k = m\n    k.w = m.w * 2.0\n    return sum(k.w)\n}\nm = M { w: zeros(2) }\nl, g = value_and_grad(f, m)", "`grad` does not differentiate through a write into a record"),
    ];
    for (src, want) in cases {
        let errs = errors(src);
        assert!(errs.iter().any(|e| e.contains(want)), "program:\n{src}\nexpected {want:?}, got {errs:?}");
    }
}

#[test]
fn the_memory_lens_shows_what_each_def_does_with_its_values() {
    // inferred from the code, never written in it
    let run = |args: &[&str]| {
        let o = Command::new(viv_bin()).current_dir(root()).args(args).output().unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        String::from_utf8_lossy(&o.stdout).into_owned()
    };
    let t = run(&["view", "lib/gpt/gpt.viv", "--lens", "memory", "--def", "optimizer_step"]);
    assert!(t.starts_with("optimizer_step(model: Gpt, g: Gpt, m: Gpt, v: Gpt, taken: number, lr_scale: number) -> nothing    lib/gpt/gpt.viv:"), "{t}");
    for line in ["  in place      model: its arrays are written, the caller sees it", "  in place      m: its arrays", "  in place      v: its arrays", "  reads         g\n"] {
        assert!(t.contains(line), "`{line}` in\n{t}");
    }
    let t = run(&["view", "lib/gpt/gpt.viv", "--lens", "memory"]);
    assert!(t.contains("  writes        m.unembed, into m, a variable of this def"), "{t}");
    assert!(t.contains("  copies        a.heads, read from a place and stored as its own value"), "{t}");
    assert!(t.contains("  loop          a pass keeps xs, ys, grad, and frees the rest of what it made"), "{t}");
    let j = run(&["view", "lib/gpt/gpt.viv", "--lens", "memory", "--def", "train", "--json"]);
    assert!(j.starts_with("{\"lens\":\"memory\",\"defs\":[{\"def\":\"train\""), "{j}");
    assert!(j.contains("{\"name\":\"model\",\"type\":\"Gpt\",\"in_place\":\"lib/gpt/gpt.viv:"), "{j}");
}
