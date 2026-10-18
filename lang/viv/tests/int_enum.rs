// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Integers, enums and the index rule, checked in all
//! three engines: the interpreter (`viv run`), Viv -> C -> native, and the
//! direct wasm backend run in node (skipped when node or a viv.wasm with the
//! runtime exports is missing; VIV_WASM=<path> names one).

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
    // under the target directory, which is inside the repository, so `#! use
    // lib/random` finds the repository root as it does for a program there
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("viv_int_enum_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// What one engine did: exit code, stdout, stderr.
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

/// A viv.wasm with the runtime exports, and node, or None.
fn wasm_runtime() -> Option<PathBuf> {
    let wasm = std::env::var("VIV_WASM").map(PathBuf::from).unwrap_or_else(|_| root().join("www/public/viv/viv.wasm"));
    let bytes = std::fs::read(&wasm).ok()?;
    if !bytes.windows(8).any(|w| w == b"vrt_init") {
        return None;
    }
    Command::new("node").arg("--version").output().ok()?;
    Some(wasm)
}

/// The three engines on `src`: (interpreter, native, direct wasm if available).
fn engines(tag: &str, src: &str) -> (Ran, Ran, Option<Ran>) {
    let dir = scratch(tag);
    let p = dir.join("p.viv");
    std::fs::write(&p, src).unwrap();
    // by absolute path, so `#! use` finds the repository root from the file (a
    // bare `p.viv` does not yet: gap R14); the diagnostics
    // still name it `p.viv`
    let i = ran(Command::new(viv_bin()).current_dir(&dir).args(["run", "-q"]).arg(&p).output().unwrap());
    let c = Command::new(viv_bin()).current_dir(&dir).arg(&p).args(["-o", "p.c"]).output().unwrap();
    assert!(c.status.success(), "{tag}: viv refused the program for C:\n{}", String::from_utf8_lossy(&c.stderr));
    let b = Command::new(cc()).current_dir(&dir).args(["-std=c99", "-O2", "p.c", "-lm", "-o", "p"]).output().unwrap();
    assert!(b.status.success(), "{tag}: the C did not build:\n{}", String::from_utf8_lossy(&b.stderr));
    let n = ran(Command::new(dir.join("p")).output().unwrap());
    let w = wasm_runtime().map(|wasm| {
        let c = Command::new(viv_bin()).current_dir(&dir).arg(&p).args(["-o", "p.wasm"]).output().unwrap();
        assert!(c.status.success(), "{tag}: viv refused the program for wasm:\n{}", String::from_utf8_lossy(&c.stderr));
        ran(Command::new("node").arg(root().join("tools/run-wasm.mjs")).arg(dir.join("p.wasm")).env("VIV_WASM", &wasm).output().unwrap())
    });
    (i, n, w)
}

/// Run `src` in every engine; each must exit 0 and print exactly `want`.
fn prints(tag: &str, src: &str, want: &str) {
    let (i, n, w) = engines(tag, src);
    assert_eq!((i.code, i.out.as_str(), i.err.as_str()), (Some(0), want, ""), "{tag}: interpreter");
    assert_eq!(i, n, "{tag}: native differs from the interpreter");
    if let Some(w) = w {
        assert_eq!((w.code, w.out.as_str()), (Some(0), want), "{tag}: direct wasm\n{}", w.err);
    }
}

/// Run `src` in every engine; each must stop with exit 1, print `out` first
/// and report an error whose first line is `error: <msg>`. The interpreter
/// and native C must agree byte for byte, the location included. The wasm
/// module, run by tools/run-wasm.mjs, prints the same output and then the message
/// without a location, as it does for an index out of bounds.
fn stops(tag: &str, src: &str, out: &str, msg: &str) {
    let (i, n, w) = engines(tag, src);
    assert_eq!(i.code, Some(4), "{tag}: interpreter {i:?}");
    assert_eq!(i.out, out, "{tag}");
    assert!(i.err.starts_with(&format!("error: {msg}\n  --> p.viv:")), "{tag}: {}", i.err);
    assert_eq!(i, n, "{tag}: native differs from the interpreter");
    if let Some(w) = w {
        assert_eq!((w.code, w.out.as_str(), w.err.as_str()), (Some(4), out, format!("error: {msg}\n").as_str()), "{tag}: direct wasm");
    }
}

/// The compiler refuses `src` with a message containing `msg`, and so does the
/// interpreter (at run time).
fn refused(tag: &str, src: &str, msg: &str) {
    let dir = scratch(tag);
    std::fs::write(dir.join("p.viv"), src).unwrap();
    let c = ran(Command::new(viv_bin()).current_dir(&dir).args(["p.viv", "-o", "p.c"]).output().unwrap());
    assert_ne!(c.code, Some(0), "{tag}: the compiler accepted it");
    assert!(c.err.contains(msg), "{tag}: compiler said\n{}", c.err);
    let i = ran(Command::new(viv_bin()).current_dir(&dir).args(["run", "-q", "p.viv"]).output().unwrap());
    assert_ne!(i.code, Some(0), "{tag}: the interpreter ran it");
    assert!(i.err.contains(msg), "{tag}: interpreter said\n{}", i.err);
}


/// Every engine stops with `error: <msg>` as the first line of what it
/// reports, after printing `out` (the interpreter and C; the wasm runner
/// prints the output too and the message without a location). For the checks C makes with no location.
fn stops_first_line(tag: &str, src: &str, out: &str, msg: &str) {
    let (i, n, w) = engines(tag, src);
    for (name, r) in [("interpreter", &i), ("native", &n)] {
        assert_eq!(r.code, Some(4), "{tag}: {name} {r:?}");
        assert_eq!(r.out, out, "{tag}: {name}");
        assert_eq!(r.err.lines().next(), Some(format!("error: {msg}").as_str()), "{tag}: {name}: {}", r.err);
    }
    if let Some(w) = w {
        assert_eq!((w.code, w.err.lines().next()), (Some(4), Some(format!("error: {msg}").as_str())), "{tag}: direct wasm");
    }
}

/// `stops_first_line` with the direct wasm engine's own first line.
fn stops_per_engine(tag: &str, src: &str, out: &str, msg: &str, wasm_msg: &str) {
    let (i, n, w) = engines(tag, src);
    for (name, r) in [("interpreter", &i), ("native", &n)] {
        assert_eq!(r.code, Some(4), "{tag}: {name} {r:?}");
        assert_eq!(r.out, out, "{tag}: {name}");
        assert_eq!(r.err.lines().next(), Some(format!("error: {msg}").as_str()), "{tag}: {name}: {}", r.err);
    }
    if let Some(w) = w {
        assert_eq!((w.code, w.err.lines().next()), (Some(4), Some(format!("error: {wasm_msg}").as_str())), "{tag}: direct wasm");
    }
}

// ------------------------------------------------------------ the index rule

#[test]
fn an_index_that_is_not_an_integer_stops_every_engine() {
    let v = "v = [10.0, 20.0, 30.0, 40.0]\n";
    stops("frac", &format!("{v}print(v[2])\nx = 2.9999999\nprint(v[x])\n"), "30\n", "index 2.9999999 is not an integer");
    stops("nan", &format!("{v}n = 0.0 / 0.0\nprint(v[n])\n"), "", "index nan is not an integer");
    stops("inf", &format!("{v}n = 1.0 / 0.0\nprint(v[n])\n"), "", "index inf is not an integer");
    // w[0.29 * 100] read w[28]: 0.29 * 100 is 28.999999999999996
    stops("product", &format!("{v}w = zeros(40)\nw[0.29 * 100] = 1.0\n"), "", "index 28.999999999999996 is not an integer");
    stops("matrix", "A = zeros(2, 3)\nprint(A[1, 0.5])\n", "", "index 0.5 is not an integer");
    stops("string", "s = \"hello\"\nprint(s[1.5])\n", "", "index 1.5 is not an integer");
    stops("list", "s = [\"a\", \"b\"]\nprint(s[0.5])\n", "", "index 0.5 is not an integer");
    // an array of records, read and written
    let recs = "record R {\n    w: Arr\n}\nrs = [R { w: zeros(1) }, R { w: zeros(2) }]\n";
    stops("recs", &format!("{recs}print(len(rs[1.5].w))\n"), "", "index 1.5 is not an integer");
    stops("recs_set", &format!("{recs}rs[0.5] = R {{ w: zeros(3) }}\n"), "", "index 0.5 is not an integer");
    // a write into a field through an element: the interpreter places the
    // error on the statement and C on the element, as for an index out of
    // bounds there (tests/objects.rs compares the first line too)
    stops_first_line("recs_field", &format!("{recs}rs[0.5].w = zeros(3)\n"), "", "index 0.5 is not an integer");
    // an integer held in a double still indexes, negative ones from the end
    prints("whole", &format!("{v}print(v[2.0])\nprint(v[-1])\nprint(v[1e0])\n"), "30\n40\n20\n");
}

// ------------------------------------------------------------ array sizes

#[test]
fn an_array_past_the_element_limit_is_refused_before_it_is_made() {
    // the C runtime multiplied the dimensions into an int: 65536 x 65537
    // wrapped to 65,536 elements and A[1, 0] wrote past the block
    let src = "print(1)\nA = zeros(65536, 65537)\nA[1, 0] = 7\nprint(A[1, 0])\n";
    // native code (the interpreter, C) holds 2^31 - 1 elements; the wasm
    // engines 16,000,000, and their refusal says so
    let wasm = " of the wasm engines (native code holds more)";
    stops_per_engine("zeros", src, "1\n", "zeros: array of 4295032832 elements exceeds the 2147483647-element limit", &format!("zeros: array of 4295032832 elements exceeds the 16000000-element limit{wasm}"));
    stops_per_engine(
        "czeros",
        "z = czeros(40000, 40000)\n",
        "",
        "czeros: complex array of 1600000000 elements exceeds the 1073741823-element limit",
        &format!("czeros: complex array of 1600000000 elements exceeds the 8000000-element limit{wasm}"),
    );
    // past the wasm engines' limit and inside native code's
    let (i, n, w) = engines("past16m", "a = zeros(20000000)\na[19999999] = 3\nprint(a[19999999] + len(a))\n");
    for (name, r) in [("interpreter", &i), ("native", &n)] {
        assert_eq!((r.code, r.out.as_str()), (Some(0), "20000003\n"), "past16m: {name} {r:?}");
    }
    if let Some(w) = w {
        assert_eq!(
            (w.code, w.err.lines().next()),
            (Some(4), Some(format!("error: zeros: array of 20000000 elements exceeds the 16000000-element limit{wasm}").as_str())),
            "past16m: direct wasm"
        );
    }
    // a negative or fractional dimension is read as the interpreter reads it
    prints("dims", "a = zeros(-3)\nprint(len(a))\nb = zeros(2.7)\nprint(len(b))\n", "0\n2\n");
}

// ------------------------------------------------------------ Int against Python

fn case_file(name: &str) -> String {
    std::fs::read_to_string(root().join("lang/viv/tests/ints").join(name)).unwrap_or_else(|e| panic!("{name}: {e} (run tools/ints/python_cases.py)"))
}

#[test]
fn every_int_case_prints_pythons_result_in_every_engine() {
    // tools/ints/python_cases.py: +, -, *, //, %, ^, abs, negation, the
    // comparisons with Ints and with numbers, conversions, text, shifts and
    // the unsigned helpers, against Python's int and NumPy's int64/uint64
    let want = case_file("python_exact.txt");
    prints("python", &case_file("python_exact.viv"), &want);
}

#[test]
fn every_int_result_that_leaves_64_bits_stops_every_engine_with_its_words() {
    for (k, line) in case_file("python_stops.tsv").lines().enumerate() {
        let (expr, msg) = line.split_once('\t').expect("expression, tab, message");
        let src = format!("def y(v: Int) -> Int {{\n    return v\n}}\nprint(\"before\")\nprint({expr})\n");
        stops(&format!("stop{k}"), &src, "before\n", msg);
    }
}

// ------------------------------------------------------------ the Int rules

#[test]
fn integer_literals_are_numbers_unless_an_int_is_expected() {
    // a literal alone is a number, as it always was: str writes "3.0"
    prints("plain", "x = 3\nprint(str(x))\nprint(x / 2)\n", "3\n1.5\n");
    // beside an Int, or bound to one, it is an Int: str writes "3"
    prints("beside", "n: Int = 3\nprint(str(n + 1))\nm = n * 2\nprint(str(m))\n", "4\n6\n");
    // `3.0` is a number even beside an Int: the sum is a number
    prints("float", "n: Int = 3\nprint(str(n + 1.0), fmt(n + 1.0))\n", "4 4.0\n");
    // len, int, floor, ceil and round adapt as literals do
    prints("adapt", "a = [1.0, 2.0, 3.0]\nk: Int = len(a) - 1\nprint(str(k))\nj: Int = floor(2.7) + round(0.4)\nprint(str(j))\n", "2\n2\n");
    // an Int bound counts an Int variable
    prints("loop", "n: Int = 3\ns: Int = 0\nfor i to n {\n    s = s + i * 10\n}\nprint(str(s))\nfor i from 1 to n {\n    show i\n}\n", "30\ni: 1\ni: 2\n");
    // `by` and `reversed` count in Ints too: the pass count of Int bounds is
    // the exact ceiling of the quotient; ceil and floor of an Int quotient are exact
    prints(
        "steps",
        "n: Int = 7\nfor i from 1 to n by 3 {\n    print(str(i))\n}\nfor j to 3 reversed {\n    print(str(j))\n}\nfor i to n reversed {\n    print(str(i))\n}\nprint(str(ceil(n / 2)), str(floor(n / 2)), str(ceil(-n / 2)))\n",
        "1\n4\n2\n1\n0\n6\n5\n4\n3\n2\n1\n0\n4 3 -3\n",
    );
}

#[test]
fn ints_and_numbers_mix_exactly_or_not_at_all() {
    // in an operator: the Int converted exactly
    prints("mix", "n: Int = 7\nprint(n * 0.5)\nprint(n / 2)\nprint(sqrt(n))\n", "3.5\n3.5\n2.64575\n");
    // compared exactly: 2^53 + 1 is not the double 2^53
    prints("cmp", "n: Int = 9007199254740993\nprint(n > 9007199254740992.0)\nprint(n == 9007199254740992.0)\n", "true\nfalse\n");
    // bound: a number where an Int is bound is refused at compile time
    refused("bind", "x: Int = 2.5\n", "`x` is an Int but is assigned a number");
    refused("rebind", "n: Int = 4\nn = n / 2\n", "`n` is an Int but is assigned a number");
    refused("param", "def f(k: Int) -> Int {\n    return k\n}\nx = 1.5\nprint(f(x))\n", "argument 1 of `f` should be an Int but is a number");
}

#[test]
fn int_arrays_and_records_hold_ints() {
    prints(
        "iarr",
        "xs: [Int] = int_zeros(3)\nxs[1] = 41\nxs[2] = xs[1] + 1\nshow xs\nprint(xs)\nprint(len(xs), xs[-1])\n",
        "xs: [0, 41, 42]\n[0, 41, 42]\n3 42\n",
    );
    // a literal where an Int array is bound; an array is a handle, as a number array is
    prints(
        "iarr_lit",
        "xs: [Int] = [1, 2, 3]\ndef f(a: [Int]) -> Int {\n    return a[0] + a[2]\n}\nprint(f(xs))\nys = xs\nys[0] = 10\nprint(xs)\n",
        "4\n[10, 2, 3]\n",
    );
    prints(
        "record",
        "record Cell {\n    i: Int\n    w: Num\n}\nc = Cell { i: 3, w: 0.5 }\nprint(c.i * 2, c.w * 2)\n",
        "6 1\n",
    );
}

#[test]
fn ints_print_their_digits() {
    prints(
        "text",
        "n: Int = 1234567\nprint(n)\nshow n\nprint(str(n), fmt(n, \"10d\"), fmt(n, \".2f\"))\nprint(f\"{n} and {n:9d}\")\n",
        "1234567\nn: 1234567\n1234567    1234567 1234567.00\n1234567 and   1234567\n",
    );
}

// ------------------------------------------------------------ enums and match

#[test]
fn enums_build_print_and_match_in_every_engine() {
    prints("enums", &case_file("enums.viv"), &case_file("enums.txt"));
}

// ------------------------------------------------------------ with records

#[test]
fn records_arrays_of_records_ints_and_enums_meet_in_every_engine() {
    prints("records", &case_file("records.viv"), &case_file("records.txt"));
    // an enum kept through a def's return and a loop, inside an array of records
    prints(
        "recs_keep",
        "enum Act {\n    Relu\n    Scale(Num)\n}\nrecord L {\n    id: Int\n    act: Act\n}\n\
         def build(n: Int) {\n    ls = []\n    for i to n {\n        act = Act.Relu\n        if i == n - 1 {\n            act = Act.Scale(2.0)\n        }\n        ls = append(ls, L { id: i, act: act })\n    }\n    return ls\n}\n\
         ls = build(3)\nk: Int = 2\nprint(ls[k].act, ls[0].act, ls[k].id)\n",
        "Scale(2) Relu 2\n",
    );
}

#[test]
fn record_operations_and_grad_refuse_int_and_enum_fields_as_they_refuse_numbers() {
    // the objects rule: arithmetic, zeros_like, for_leaves and grad act on a
    // record of arrays; an Int or an enum field is refused as a Num field is
    let rec = "enum K {\n    A\n    B\n}\nrecord P {\n    w: Arr\n    n: Int\n}\nrecord Q {\n    w: Arr\n    k: K\n}\np = P { w: zeros(2), n: 3 }\nq = Q { w: zeros(2), k: K.A }\n";
    refused("rec_add", &format!("{rec}r = p + p\n"), "`+` over a record acts on its arrays, and the field `n` of `P` is an Int");
    refused("rec_zeros", &format!("{rec}r = zeros_like(p)\n"), "`zeros_like` over a record acts on its arrays, and the field `n` of `P` is an Int");
    refused("rec_enum", &format!("{rec}r = q * 2.0\n"), "`*` over a record acts on its arrays, and the field `k` of `Q` is a K");
    refused(
        "rec_grad",
        &format!("{rec}def loss(p: P) {{\n    return sum(p.w * p.w)\n}}\nv, g = value_and_grad(loss, p)\n"),
        "`grad` differentiates a record of arrays, and the field `n` of `P` is a value grad does not follow",
    );
}

#[test]
fn an_array_of_enums_is_refused_cleanly() {
    refused("enum_arr_lit", "enum K {\n    A\n    B\n}\nks = [K.A, K.B]\n", "an array of enum values (`[K]`) is not supported yet");
    refused("enum_arr_ty", "enum K {\n    A\n    B\n}\nks: [K] = []\n", "an array of enum values (`[K]`) is not supported yet");
    refused("unknown_list", "ks: [Nope] = []\n", "unknown type `Nope`");
}

#[test]
fn a_match_must_cover_every_variant() {
    let decl = "enum Shape {\n    Dot\n    Circle(Num)\n    Box(Num, Num)\n}\ns = Shape.Circle(2.0)\n";
    refused("missing", &format!("{decl}match s {{\n    Dot => print(0)\n    Circle(r) => print(r)\n}}\n"), "this `match` does not cover `Box`");
    refused("twice", &format!("{decl}match s {{\n    Dot => print(0)\n    Dot => print(1)\n    _ => print(2)\n}}\n"), "this arm can never run: `Dot` is matched by an arm before it");
    refused("after_wild", &format!("{decl}match s {{\n    _ => print(0)\n    Dot => print(1)\n}}\n"), "this arm can never run: the arm `_` before it takes every value");
    refused("unknown", &format!("{decl}match s {{\n    Square(a) => print(a)\n    _ => print(1)\n}}\n"), "`Shape` has no variant `Square`");
    refused("arity", "enum E {\n    A(Num)\n}\nx = E.A(1.0, 2.0)\n", "`E.A` holds 1 value, and 2 are given");
    refused("payload", "enum E {\n    A(Int)\n}\nx = E.A(1.5)\n", "`E.A` holds an Int here, and this is a number");
    refused("eq", &format!("{decl}t = Shape.Dot\nprint(s == t)\n"), "`==` compares enums without payloads, and `Shape` has them");
    refused("none", "x = None\n", "`None` of what type?");
    // a binder belongs to its arm: two arms bind `v` to values of two types
    prints(
        "binders",
        "enum V {\n    I(Int)\n    S(Str)\n}\ndef say(x: V) {\n    match x {\n        I(v) => print(v + 1)\n        S(v) => print(v + \"!\")\n    }\n}\nsay(V.I(41))\nsay(V.S(\"hi\"))\n",
        "42\nhi!\n",
    );
}

#[test]
fn an_enum_value_is_kept_as_long_as_it_is_reachable() {
    // a list built in a loop and through a def, a record holding one, a
    // value shared by two others, garbage made each pass; the C and wasm
    // runtimes free and keep nodes on their arenas
    prints(
        "lists",
        "enum List {\n    Nil\n    Cons(Int, List)\n}\n\
         def push(xs: List, v: Int) -> List {\n    junk = List.Cons(v * 2, List.Nil)\n    return List.Cons(v, xs)\n}\n\
         def total(xs: List) -> Int {\n    s: Int = 0\n    cur = xs\n    going = true\n    while going {\n        match cur {\n            Nil => going = false\n            Cons(v, rest) => {\n                s = s + v\n                cur = rest\n            }\n        }\n    }\n    return s\n}\n\
         xs = List.Nil\nn: Int = 3000\nfor i to n {\n    xs = push(xs, i)\n    tmp = List.Cons(i, List.Nil)\n}\nprint(total(xs))\n\
         shared = List.Cons(5, List.Nil)\npair = List.Cons(1, shared)\nother = List.Cons(2, shared)\nprint(pair, other)\n\
         record Holder {\n    name: Str\n    item: List\n}\nh = Holder { name: \"h\", item: push(List.Nil, 7) }\nprint(h.item)\n",
        "4498500\nCons(1, Cons(5, Nil)) Cons(2, Cons(5, Nil))\nCons(7, Nil)\n",
    );
}

#[test]
fn a_recursive_def_returns_ints_and_optionals_as_declared() {
    // the recursion limit's rewrite holds each return value in a temporary;
    // it takes the declared type, so `return 1` is an Int there too
    prints(
        "rec_int",
        "def fact(n: Int) -> Int {\n    if n <= 1 {\n        return 1\n    }\n    return n * fact(n - 1)\n}\nprint(fact(20))\n",
        "2432902008176640000\n",
    );
    prints(
        "rec_opt",
        "def find(xs: [Int], k: Int, at: Int) -> Int? {\n    if at == len(xs) {\n        return None\n    }\n    if xs[at] == k {\n        return Some(at)\n    }\n    return find(xs, k, at + 1)\n}\nxs: [Int] = [4, 7, 9]\nprint(find(xs, 9, 0), find(xs, 5, 0))\n",
        "Some(2) None\n",
    );
}

#[test]
fn a_def_that_can_reach_its_end_without_a_value_is_refused_unless_it_returns_a_number() {
    // a number-returning def gives 0 there in every engine, C included
    prints("falls_num", "def h(n) {\n    if n > 2 {\n        return n\n    }\n}\nprint(h(1), h(5))\n", "0 5\n");
    refused("falls_int", "def g(n: Int) -> Int {\n    if n > 2 {\n        return n\n    }\n}\nprint(g(1))\n", "`g` returns an Int but can reach its end without a `return`");
    refused("falls_str", "def s(n) {\n    if n > 2 {\n        return \"x\"\n    }\n}\nprint(s(1))\n", "`s` returns a string but can reach its end without a `return`");
    refused("falls_arr", "def a(n) {\n    if n > 2 {\n        return zeros(2)\n    }\n}\nprint(a(1))\n", "`a` returns an array but can reach its end without a `return`");
    // every way out returns, stops, or loops forever: accepted
    prints(
        "ends",
        "def a(n: Int) -> Int {\n    if n > 2 {\n        return n\n    } else {\n        return 0\n    }\n}\n\
         def b(n: Int) -> Int {\n    while true {\n        if n > 5 {\n            return n\n        }\n        n = n + 1\n    }\n}\n\
         def c(n: Int) -> Int {\n    if n > 2 {\n        return n\n    }\n    stop(\"too small\")\n}\n\
         enum E {\n    X\n    Y\n}\ndef d(e: E) -> Int {\n    match e {\n        X => return 1\n        Y => return 2\n    }\n}\n\
         print(a(1), b(1), c(3), d(E.Y))\n",
        "0 6 3 2\n",
    );
}

#[test]
fn an_unpacked_int_stays_an_int_under_mod() {
    // B304 (vivsmith): `b % 5` was typed a number while b's type was still
    // unknown (it comes from the multiple return's record), and stayed one
    prints(
        "unpacked_int_mod",
        "def f(p: Int) {\n    return 1.0, p\n}\nz: Int = 3\na, b = f(z)\nb = b % 2\nc, d = f(b)\nb2 = d // 1\nc2, d2 = f(b2)\nprint(fmt(a), b, d, d2)\n",
        "1.0 1 1 1\n",
    );
}

#[test]
fn abs_min_max_of_whole_numbers_and_int_slots_are_ints() {
    // B305 (vivsmith): abs, min and max of int(...), len(...) and literals
    // were numbers, and so was a returned slot holding int(...)
    prints(
        "abs_min_max_int",
        "def g(x: Num) {\n    return x, int(x)\n}\ndef h(x: Num) {\n    if x > 1.0 {\n        return x, int(x)\n    }\n    return x, 0.5\n}\n\
         x = 2.5\na = [1.0, 2.0]\nm1: Int = abs(int(-x))\nm2: Int = min(int(x), 9)\nm3: Int = max(len(a), -4)\nm4: Int = abs(-5)\n\
         y, m5 = g(x)\nn: Int = m5\nu, w = h(0.5)\nk = abs(int(x)) + min(int(x), 7)\nk2: Int = k\nq = abs(int(x)) * 0.5\n\
         print(m1, m2, m3, m4, n, fmt(w), k2, fmt(q))\n",
        "2 2 2 5 2 0.5 4 1.0\n",
    );
}
