// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The Python-like surface: comprehensions, `a if c else b`, negative indices,
//! `in` on arrays and maps, `key=`, `enumerate(xs, start)`, `zip`, `rows(M)`
//! and `print`'s separator (crate::surface, parser.rs).
//!
//! Every case of tools/surface/cases.json runs in the interpreter, as C built
//! natively and (when node and a viv.wasm with the runtime are present) as a
//! module from the direct wasm backend, and must print what CPython printed
//! for the Python program it stands for, byte for byte, in all three. A case
//! that stops must stop with Viv's words in every engine, after the same
//! output. `python3 tools/surface/cases.py` regenerates the file.

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
    let dir = std::env::temp_dir().join(format!("viv_surface_{}_{tag}", std::process::id()));
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
    Ran {
        code: o.status.code(),
        out: String::from_utf8_lossy(&o.stdout).into_owned(),
        err: String::from_utf8_lossy(&o.stderr).into_owned(),
    }
}

fn interp(dir: &Path) -> Ran {
    ran(Command::new(viv_bin()).current_dir(dir).args(["run", "-q", "p.viv"]).output().unwrap())
}

fn native(dir: &Path) -> Option<Ran> {
    if Command::new(cc()).arg("--version").output().is_err() {
        return None;
    }
    let c = Command::new(viv_bin()).current_dir(dir).args(["p.viv", "-o", "p.c"]).output().unwrap();
    assert!(c.status.success(), "viv refused the program:\n{}", String::from_utf8_lossy(&c.stderr));
    let b = Command::new(cc()).current_dir(dir).args(["-std=c99", "-O0", "-Wall", "p.c", "-lm", "-o", "p"]).output().unwrap();
    assert!(b.status.success(), "the C did not build:\n{}", String::from_utf8_lossy(&b.stderr));
    Some(ran(Command::new(dir.join("p")).output().unwrap()))
}

/// The direct wasm backend in node, or None when there is no runtime to run it.
fn direct(dir: &Path) -> Option<Ran> {
    let wasm = std::env::var("VIV_WASM").map(PathBuf::from).unwrap_or_else(|_| root().join("www/public/viv/viv.wasm"));
    let bytes = std::fs::read(&wasm).ok()?;
    if !bytes.windows(12).any(|w| w == b"vrt_getslice") {
        eprintln!("skip wasm: {} has no slice runtime (www/build-wasm.sh builds one)", wasm.display());
        return None;
    }
    Command::new("node").arg("--version").output().ok()?;
    let c = Command::new(viv_bin()).current_dir(dir).args(["p.viv", "-o", "p.wasm"]).output().unwrap();
    assert!(c.status.success(), "the wasm backend refused the program:\n{}", String::from_utf8_lossy(&c.stderr));
    Some(ran(Command::new("node").arg(root().join("tools/run-wasm.mjs")).arg(dir.join("p.wasm")).env("VIV_WASM", &wasm).output().unwrap()))
}

/// Run `src` in every engine; they must agree. Returns the interpreter's run.
fn all(tag: &str, src: &str) -> Ran {
    let dir = scratch(tag);
    std::fs::write(dir.join("p.viv"), src).unwrap();
    let i = interp(&dir);
    if let Some(n) = native(&dir) {
        if i.code == Some(0) {
            assert_eq!(i, n, "interpreter and C disagree on\n{src}");
        } else {
            assert_eq!((&n.code, &n.out), (&i.code, &i.out), "{tag}: interpreter and C stop differently on\n{src}");
            assert_eq!(n.err.lines().next(), i.err.lines().next(), "{tag}: interpreter and C stop differently on\n{src}");
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
    i
}

/// The interpreter's refusal of `src` (the compiler must refuse it too).
fn refused(tag: &str, src: &str) -> String {
    let dir = scratch(tag);
    std::fs::write(dir.join("p.viv"), src).unwrap();
    let c = ran(Command::new(viv_bin()).current_dir(&dir).args(["p.viv", "-o", "p.c"]).output().unwrap());
    assert_ne!(c.code, Some(0), "{tag}: the compiler accepted it");
    let i = ran(Command::new(viv_bin()).current_dir(&dir).args(["run", "-q", "p.viv"]).output().unwrap());
    assert_ne!(i.code, Some(0), "{tag}: the interpreter ran it");
    i.err
}

#[test]
fn every_case_prints_what_cpython_prints() {
    let text = std::fs::read_to_string(root().join("tools/surface/cases.json")).unwrap();
    let json = viv::story::parse_json(&text).unwrap();
    let viv::story::Json::Arr(cases) = json.get("cases").unwrap() else { panic!("cases") };
    assert!(cases.len() >= 20, "{} cases", cases.len());
    for c in cases {
        let name = c.get("name").and_then(|v| v.str()).unwrap();
        let src = c.get("viv").and_then(|v| v.str()).unwrap();
        let want = c.get("out").and_then(|v| v.str()).unwrap();
        let r = all(name, src);
        assert_eq!(r.out, want, "{name}: not what CPython printed");
        match c.get("stops").and_then(|v| v.str()) {
            None => assert_eq!(r.code, Some(0), "{name}: {r:?}"),
            Some(words) => {
                assert_eq!(r.code, Some(4), "{name}: {r:?}");
                assert!(r.err.starts_with(words), "{name}: stopped with\n{}", r.err);
            }
        }
    }
}

#[test]
fn show_of_a_comprehension_has_no_label() {
    let r = all("showcomp", "xs = [1, 2]\nshow [x * 2 for x in xs]\nshow xs\n");
    assert_eq!(r.out, "[2, 4]\nxs: [1, 2]\n");
}

#[test]
fn a_program_without_the_new_forms_parses_as_before() {
    // the pass leaves a program with no comprehension, if-expression or key=
    // untouched: the AST the parser returns is the one it returned before
    let src = "def f(x) {\n    return x * 2\n}\nfor i, c in enumerate(\"ab\") {\n    print(i, c)\n}\nfor x in [1, 2] {\n    show f(x)\n}\n";
    let a = format!("{:?}", viv::parse(src, knotkit::span::SourceId(0)).unwrap());
    assert!(!a.contains("viv__lift") && !a.contains("viv__if") && !a.contains("viv__cmp"), "{a}");
}

#[test]
fn refusals_are_located() {
    let e = refused("noelse", "x = 1 if 2 > 1\n");
    assert!(e.contains("an `if` expression needs `else`"), "{e}");
    let e = refused("lambda", "def mk(a) {\n    return fn(x) -> [a * y for y to x]\n}\nshow mk(2.0)\n");
    assert!(e.contains("it cannot be returned or stored") && e.contains("p.viv:2:12"), "{e}");
    let e = refused("end", "print(1, end=\"\")\n");
    assert!(e.contains("`print` has no `end=`"), "{e}");
    let e = refused("kw", "x = len([1], key=abs)\n");
    assert!(e.contains("unknown keyword argument `key=`") && e.contains("`len` takes no keyword arguments"), "{e}");
    // a def's parameters may be named at a call; a
    // name it does not have is refused, listing the ones it has
    let e = refused("owndef", "def f(a) {\n    return a\n}\nx = f(b=1)\n");
    assert!(e.contains("`f` has no parameter `b`") && e.contains("`f(a)`"), "{e}");
    let e = refused("zipnames", "for a, b in zip([1], [2], [3]) {\n    show a\n}\n");
    assert!(e.contains("`zip` of 3 sequences gives 3 names"), "{e}");
    let e = refused("pair", "for i, x in [1, 2] {\n    show x\n}\n");
    assert!(e.contains("takes `enumerate(e)` or `zip(a, b, ...)`"), "{e}");
    let e = refused("range", "print([i for i in range(3)])\n");
    assert!(e.contains("expected `to`"), "{e}");
    let e = refused("reverse", "print(sorted([2, 1], reverse=1))\n");
    assert!(e.contains("`reverse=` is `true` or `false`"), "{e}");
}

#[test]
fn comprehensions_inside_lambdas_with_captures() {
    // a lambda's body may hold a comprehension, a map
    // comprehension, a generator argument or an if-expression, and it reads
    // the locals of its def by value, in all three engines.
    let src = "def apply(f, x) {\n    return f(x)\n}\n\
vocab = [\"h\", \"e\", \"l\", \"o\"]\n\
stoi = {c: i for i, c in enumerate(vocab)}\n\
encode = fn(text) -> [stoi[c] for c in text]\n\
decode = fn(ids) -> join(\"\", [vocab[i] for i in ids])\n\
print(encode(\"hello\"), decode(encode(\"hello\")))\n\
def run(a, xs: Arr) {\n\
    total = fn(k) -> sum(a * x + k for x in xs)\n\
    index = fn(ws) -> {w: i * a for i, w in enumerate(ws)}\n\
    clip = fn(x) -> x if x < a else a\n\
    a = 100.0\n\
    return total(1.0) + apply(total, 0.0) + index([\"p\", \"q\"])[\"q\"] + clip(5.0)\n\
}\n\
print(run(2.0, [1.0, 2.0, 3.0]))\n";
    let r = all("lambda_lift", src);
    assert_eq!(r.code, Some(0), "{r:?}");
    assert_eq!(r.out, "[0, 1, 2, 2, 3] hello\n31\n", "{r:?}");
}

#[test]
fn record_defaults_and_constructor_defs() {
    // defaults fill a literal, an update takes the rest
    // from its base, a constructor def is called by the record's name, and a
    // comprehension of constructor calls draws in order, in all three engines
    let src = "LIMIT: Int = 16\n\
record Cfg { n: Int, k: Int = 8, w: Int = LIMIT, s: Num = 1.0 / 4.0 }\n\
record Pair { a: Arr, b: Arr }\n\
def Pair(n: Int) -> Pair {\n    return Pair { a: randn(n), b: randn(n) }\n}\n\
c = Cfg { n: 9 }\n\
u = Cfg { k: 2, ..c }\n\
srand(5)\n\
ps = [Pair(2) for i to 2]\n\
srand(5)\n\
first = randn(2)\n\
print(c.k, c.w, c.s, u.k, u.w, ps[0].a[1] == first[1], len(ps))\n";
    let r = all("ctor_defaults", src);
    assert_eq!(r.code, Some(0), "{r:?}");
    assert_eq!(r.out, "8 16 0.25 2 16 true 2\n", "{r:?}");
}

#[test]
fn record_refusals_are_located() {
    let e = refused("ctor_default_call", "record P { a: Num, b: Num = rand() }\nshow P { a: 1.0 }.a\n");
    assert!(e.contains("the default of `b` is not a constant: it calls `rand`") && e.contains("p.viv:1:29"), "{e}");
    let e = refused("ctor_default_field", "record P { a: Num, b: Num = a }\nshow P { a: 1.0 }.a\n");
    assert!(e.contains("reads `a`, which is not a top-level constant"), "{e}");
    let e = refused("ctor_ret", "record P { a: Num }\ndef P(x) {\n    return P { a: x }\n}\nshow P(1.0).a\n");
    assert!(e.contains("`P` is the constructor of record `P`, so it returns a `P`") && e.contains("p.viv:2:5"), "{e}");
    let e = refused("ctor_none", "record P { a: Num }\nshow P(1.0).a\n");
    assert!(e.contains("`P` is a record with no constructor def") && e.contains("p.viv:2:6"), "{e}");
    let e = refused("ctor_order", "record P { a: Num, b: Num }\np = P { b: rand(), a: rand() }\nshow p.a\n");
    assert!(e.contains("run in the order `P` declares its fields") && e.contains("p.viv:2:20"), "{e}");
    let e = refused("ctor_missing", "record P { a: Num, b: Num }\np = P { a: 1.0 }\nshow p.a\n");
    assert!(e.contains("missing field `b`"), "{e}");
}
