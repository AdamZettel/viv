// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Strings, in all three engines: the interpreter (`viv
//! run`), Viv -> C -> native, and the direct wasm backend run in node (skipped,
//! with a note, when node or a viv.wasm with the runtime exports is missing;
//! VIV_WASM=<path> names one).
//!
//! A program that runs must print the same bytes in every engine; one that
//! stops must stop with the same words, and the interpreter and C at the same
//! location. `cases_match_cpython` runs tools/strings/cases.viv, whose
//! expected output tools/strings/cases.py computed with CPython.

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
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("viv_strings_{}_{tag}", std::process::id()));
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

fn wasm_runtime() -> Option<PathBuf> {
    let wasm = std::env::var("VIV_WASM").map(PathBuf::from).unwrap_or_else(|_| root().join("www/public/viv/viv.wasm"));
    let bytes = std::fs::read(&wasm).ok()?;
    if !bytes.windows(10).any(|w| w == b"vrt_s_find") {
        eprintln!("note: {} has no string runtime; the direct wasm engine is skipped", wasm.display());
        return None;
    }
    Command::new("node").arg("--version").output().ok()?;
    Some(wasm)
}

/// The three engines on `src` (with `files` beside it): interpreter, native, direct wasm.
fn engines_with(tag: &str, src: &str, files: &[(&str, &str)]) -> (Ran, Ran, Option<Ran>) {
    engines_opt(tag, src, files, "-O2")
}

/// `engines_with`, the C built at optimization `opt`.
fn engines_opt(tag: &str, src: &str, files: &[(&str, &str)], opt: &str) -> (Ran, Ran, Option<Ran>) {
    let dir = scratch(tag);
    let p = dir.join("p.viv");
    std::fs::write(&p, src).unwrap();
    for (name, text) in files {
        let f = dir.join(name);
        std::fs::create_dir_all(f.parent().unwrap()).unwrap();
        std::fs::write(f, text).unwrap();
    }
    let i = ran(Command::new(viv_bin()).current_dir(&dir).args(["run", "-q"]).arg(&p).output().unwrap());
    let c = Command::new(viv_bin()).current_dir(&dir).arg(&p).args(["-o", "p.c"]).output().unwrap();
    assert!(c.status.success(), "{tag}: viv refused the program for C:\n{}", String::from_utf8_lossy(&c.stderr));
    let b = Command::new(cc()).current_dir(&dir).args(["-std=c99", opt, "p.c", "-lm", "-o", "p"]).output().unwrap();
    assert!(b.status.success(), "{tag}: the C did not build:\n{}", String::from_utf8_lossy(&b.stderr));
    let n = ran(Command::new(dir.join("p")).current_dir(&dir).output().unwrap());
    let w = wasm_runtime().map(|wasm| {
        let c = Command::new(viv_bin()).current_dir(&dir).arg(&p).args(["-o", "p.wasm"]).output().unwrap();
        assert!(c.status.success(), "{tag}: viv refused the program for wasm:\n{}", String::from_utf8_lossy(&c.stderr));
        let mut cmd = Command::new("node");
        cmd.current_dir(&dir).arg(root().join("tools/run-wasm.mjs"));
        for (name, _) in files {
            cmd.arg(format!("--file={name}={}", dir.join(name).display()));
        }
        ran(cmd.arg(dir.join("p.wasm")).env("VIV_WASM", &wasm).output().unwrap())
    });
    (i, n, w)
}

fn engines(tag: &str, src: &str) -> (Ran, Ran, Option<Ran>) {
    engines_with(tag, src, &[])
}

/// Every engine exits 0 and prints exactly `want`.
fn prints(tag: &str, src: &str, want: &str) {
    let (i, n, w) = engines(tag, src);
    assert_eq!((i.code, i.out.as_str(), i.err.as_str()), (Some(0), want, ""), "{tag}: interpreter");
    assert_eq!(i, n, "{tag}: native differs from the interpreter");
    if let Some(w) = w {
        assert_eq!((w.code, w.out.as_str()), (Some(0), want), "{tag}: direct wasm\n{}", w.err);
    }
}

/// Every engine stops with `error: <msg>` after printing `out`; the
/// interpreter and C agree byte for byte, the location included.
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

/// The compiler refuses `src` with a message containing `msg`, and so does `viv run`.
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

// ---- CPython's answers ------------------------------------------------------------

#[test]
fn cases_match_cpython() {
    let src = std::fs::read_to_string(root().join("tools/strings/cases.viv")).unwrap();
    let want = std::fs::read_to_string(root().join("tools/strings/cases.out")).unwrap();
    // one main of 58,000 IR nodes: cc -O2 takes minutes on it, -O0 seconds
    let (i, n, w) = engines_opt("cases", &src, &[], "-O0");
    let first_diff = |got: &str| {
        got.lines().zip(want.lines()).enumerate().find(|(_, (a, b))| a != b).map(|(k, (a, b))| format!("line {}: got {a:?}, CPython {b:?}", k + 1))
    };
    assert_eq!(i.err, "", "interpreter");
    assert!(i.out == want, "interpreter: {:?}", first_diff(&i.out));
    assert!(n.out == want, "native: {:?} {}", first_diff(&n.out), n.err);
    if let Some(w) = w {
        assert!(w.out == want, "direct wasm: {:?} {}", first_diff(&w.out), w.err);
    }
}

// ---- the surface --------------------------------------------------------------------

#[test]
fn literals_and_escapes() {
    prints(
        "lit",
        "print(\"tab\\there\")\nprint(\"q\\\"uote \\\\ back\")\nprint(\"\\u{e9}\\u{1F600} \\u{3a3}\")\nprint(\"a'b\")\nshow \"it's\"\nshow \"x\\ny\\r\\0\"\nprint(len(\"\"), len(\"\\u{1F600}\"))\n",
        "tab\there\nq\"uote \\ back\né😀 Σ\na'b\n\"it's\"\n'x\\ny\\r\\x00'\n0 1\n",
    );
    refused("escape", "print(\"a\\qb\")\n", "unknown escape `\\q`");
    refused("surrogate", "print(\"\\u{d800}\")\n", "is not a Unicode character");
}

#[test]
fn the_design_example() {
    prints(
        "design",
        r#"name = "water"
line = "O  0.000  0.000  0.117"
parts = split(line)
z = float(parts[3])
print(upper(name), "has", len(parts) - 1, "coordinates")
label = f"{name}: z = {z:.3f}"
if startswith(line, "O") and "0.117" in line {
    show label
}
for c in "héllo" {
    print(c)
}
words = []
for w in split("a b c") {
    words = append(words, upper(w))
}
print(join("-", words))
print(line.split()[0].lower())
show words
"#,
        "WATER has 3 coordinates\nlabel: 'water: z = 0.117'\nh\né\nl\nl\no\nA-B-C\no\nwords: ['A', 'B', 'C']\n",
    );
}

#[test]
fn fstrings() {
    prints(
        "fstr",
        r#"x = 0.1 + 0.2
n = 42
s = "héllo"
print(f"x = {x}, {x:.3f}, {x=}, {x = :.2e}")
print(f"{n:5d}|{s:>8}|{s:*<8}|{s!r}|{s=}")
print(f"{{braces}} {f"{s[0]}{s[-1]}"} {join(", ", ["a", "b"])}")
print(f"{len(s) * 2} {upper(s)!r:^11}|")
t = f"no fields"
show t
"#,
        "x = 0.30000000000000004, 0.300, x=0.30000000000000004, x = 3.00e-01\n   42|   héllo|héllo***|'héllo'|s='héllo'\n{braces} ho a, b\n10   'HÉLLO'  |\nt: 'no fields'\n",
    );
    refused("fstr-brace", "print(f\"a } b\")\n", "a single `}` is not allowed");
    refused("fstr-empty", "print(f\"a {} b\")\n", "needs an expression");
}

#[test]
fn indexing_slicing_and_iteration() {
    prints(
        "index",
        r#"s = "héllo wörld"
print(f"{len(s):d} {s[0]}{s[1]}{s[-1]} {s[1:4]} {s[::-1]} {s[-5:]} {s[:80]} [{s[20:]}]")
k = 0
for i, c in enumerate(s) {
    if c == "ö" {
        k = i
    }
}
print(k, s[k])
w = ["x", "yy", "zzz"]
for x in w {
    print(x, len(x))
}
print(w[-1], repr(w[::-1]), repr(w[1:]))
total = 0.0
for v in [1.5, 2.5] {
    total += v
}
show total
"#,
        "11 héd éll dlröw olléh wörld héllo wörld []\n7 ö\nx 1\nyy 2\nzzz 3\nzzz ['zzz', 'yy', 'x'] ['yy', 'zzz']\ntotal: 4\n",
    );
}

#[test]
fn defs_records_tuples_and_globals() {
    prints(
        "defs",
        r#"record Atom { sym: Str, x: Num }
def label(a: Atom) -> Str {
    return f"{a.sym}@{a.x}"
}
def pieces(line) {
    p = split(line, ",")
    return p[0], float(p[1])
}
def shout(s) {
    return upper(s) + "!"
}
def names(n) {
    out = []
    for i to n {
        out = append(out, chr(65 + i))
    }
    return out
}
a = Atom { sym: "He", x: 2.5 }
print(label(a))
sym, x = pieces("Ne,20.18")
print(sym, x)
print(shout("hi"), shout(sym))
print(repr(names(4)), join("", names(3)))
greeting = "hello"
def greet() {
    return greeting + ", world"
}
print(greet())
"#,
        "He@2.5\nNe 20.18\nHI! NE!\n['A', 'B', 'C', 'D'] ABC\nhello, world\n",
    );
}

#[test]
fn methods_and_membership() {
    prints(
        "methods",
        r#"line = "  Alpha, Beta ,gamma  "
parts = line.strip().split(",")
for p in parts {
    print("[", p.strip().lower(), "]", sep="")
}
print("beta" in line.lower(), "x" not in line, "Beta" in ["Alpha", "Beta"])
print(line.count("a"), line.find("Beta"), line.rfind("a"), line.replace(" ", ""))
print("3".zfill(4), " ", "ab".center(6, "*"), " ", "é".ljust(3), "|", sep="")
print("42".isdigit(), "".isalpha(), "ΑΒΓ".isupper(), "Straße".upper())
"#,
        "[alpha]\n[beta]\n[gamma]\ntrue true true\n4 9 19 Alpha,Beta,gamma\n0003 **ab** é  |\ntrue false true STRASSE\n",
    );
}

#[test]
fn numbers_from_and_to_text() {
    prints(
        "numbers",
        r#"print(repr(float("1_000.5")), repr(float(" -1e-3 ")), repr(float("inf")), repr(float("nan")))
print(repr(int("0x1f", 16)), repr(int(" -42 ")), repr(int("2" * 20, 3)), repr(int(-2.7)))
print(str(3.0), str(0.1 + 0.2), str(true), str([1.0, 2.5]), repr("a\tb"))
print(fmt(2.5, ".3f"), " ", fmt(1234.5, "10.2e"), "|", fmt("ab", ">5"), "|", fmt(["a"], ">7"), "|", sep="")
print(ord("€"), chr(8364), repr(float("1e400")), repr(float("2.4703282292062328e-324")))
"#,
        "1000.5 -0.001 inf nan\n31.0 -42.0 3486784400.0 -2.0\n3 0.30000000000000004 true [1, 2.5] 'a\\tb'\n2.500   1.23e+03|   ab|  ['a']|\n8364 € inf 5e-324\n",
    );
}

#[test]
fn comparisons_order_by_code_point() {
    prints(
        "compare",
        "print(\"a\" < \"b\", \"B\" < \"a\", \"é\" > \"z\", \"ab\" < \"abc\", \"\" == \"\", \"x\" != \"x\")\nprint(repr(sorted([\"é\", \"z\", \"a\", \"B\", \"\"])))\nprint([\"a\", \"b\"] == [\"a\", \"b\"], [\"a\"] != [\"a\", \"b\"])\n",
        "true true true true true false\n['', 'B', 'a', 'z', 'é']\ntrue true\n",
    );
}

#[test]
fn a_list_grows_in_place_and_an_alias_keeps_its_copy() {
    prints(
        "grow",
        r#"a = ["x"]
b = a
a = append(a, "y")
a = extend(a, ["z", "w"])
print(repr(a), repr(b))
words = []
for i to 5000 {
    words = append(words, fmt(i, "d"))
}
print(len(words), words[4999], len(join(",", words)))
"#,
        "['x', 'y', 'z', 'w'] ['x']\n5000 4999 23889\n",
    );
}

#[test]
fn loops_that_make_strings_free_what_they_dropped() {
    // ten thousand lines built and dropped: every engine keeps only the last
    prints(
        "loopfree",
        r#"last = ""
count = 0.0
for i to 10000 {
    line = f"{i:d}: {repeat_it(i)}"
    if endswith(line.split(":")[0], "7") {
        count += 1
    }
    last = line
}
def repeat_it(i) {
    return "ab" * (i % 5)
}
print(count, last)
"#,
        "1000 9999: abababab\n",
    );
}

#[test]
fn figures_and_animation_take_string_values() {
    prints(
        "figure",
        r#"name = "E(k)"
figure(f"band of {name}", "k", "E")
plot([0.0, 1.0], [1.0, 2.0], "b")
cap = "step " + str(3.0)
say(cap)
say(cap, 2.5)
"#,
        "figure: band of E(k) ; k ; E\nplot: b ; [0, 1] ; [1, 2]\nsay: step 3\nsay: step 3 ; 2.5\n",
    );
    stops("anim-nl", "t = \"a\\nb\"\nsay(t)\n", "", "text in an animation cannot contain a newline or ` ; `");
    // a diagram's note has three texts (id, text, style), all of them values here
    prints(
        "note",
        r#"def label(id: Str, color: Str, t) {
    note("t" + id, "T = " + fmt(t, ".1f") + " N", color + " left bold", 10.0, 20.0)
}
figure("d", "", "", 0, 100, 0, 50, "diagram")
label("1", "s2", 84.87)
"#,
        "figure: d ;  ;  ; 0 100 0 50 ; diagram\nnote: t1 ; T = 84.9 N ; s2 left bold ; 10 20\n",
    );
}

// ---- stops, located -------------------------------------------------------------------

#[test]
fn errors_stop_with_pythons_words() {
    stops("err-index", "s = \"abc\"\nprint(s[0])\nprint(s[3])\n", "a\n", "string index out of range");
    stops("err-lindex", "w = split(\"a b\")\nprint(w[-3])\n", "", "list index out of range");
    stops("err-float", "print(1)\nx = float(\"1e\")\n", "1\n", "could not convert string to float: '1e'");
    stops("err-int", "x = int(\"1.5\")\n", "", "invalid literal for int() with base 10: '1.5'");
    stops("err-chr", "x = chr(-1)\n", "", "chr() arg not in range(0x110000)");
    stops("err-ord", "x = ord(\"ab\")\n", "", "ord() expected a character, but string of length 2 found");
    stops("err-sep", "x = split(\"a b\", \"\")\n", "", "empty separator");
    stops("err-notfound", "x = index(\"abc\", \"z\")\n", "", "substring not found");
    stops("err-fill", "x = ljust(\"a\", 3, \"ab\")\n", "", "The fill character must be exactly one character long");
    stops("err-stop", "who = \"x\"\nstop(f\"no atom {who}\")\n", "", "no atom x");
    stops("err-step", "s = \"abc\"\nk = 0\nprint(s[::k])\n", "", "slice step cannot be zero");
    stops("err-listnot", "x = index([\"a\"], \"b\")\n", "", "'b' is not in list");
}

#[test]
fn refusals_are_located() {
    refused("setchar", "s = \"abc\"\ns[0] = \"x\"\n", "'str' object does not support item assignment");
    refused("setlist", "w = [\"a\"]\nw[0] = \"x\"\n", "a list of strings cannot be changed in place");
    refused("minus", "x = \"a\" - \"b\"\n", "`-` does not apply to a string and a string");
    refused("mix", "x = \"a\" + 1\n", "`+` does not apply to a string and a number");
    refused("mixlist", "x = [\"a\", 1]\n", "a list of strings cannot hold a number");
    refused("inarr", "x = [1.0] in [1.0, 2.0]\n", "`in` looks for a number in an array");
    refused("innum", "x = 1.0 in 2.0\n", "`in` takes a string, a list of strings, a real array or a map on its right");
    refused("usage", "x = find(\"a\")\n", "`find` takes find(string, string, [number], [number])");
    refused("parallel", "parallel for i to 3 {\n    s = str(i)\n}\n", "a `parallel for` cannot make strings");
    refused("rangeloop", "for i in range(3) {\n    show i\n}\n", "a Viv loop is `for i to n { ... }`");
}

// ---- files --------------------------------------------------------------------------

#[test]
fn text_files_read_and_written() {
    let src = "t = read_text(\"in.txt\")\nprint(repr(t))\nn = write_text(\"out.txt\", upper(t))\nprint(n, repr(read_lines(\"out.txt\")))\n";
    let (i, n, w) = engines_with("files", src, &[("in.txt", "a\r\nb é\rc\n")]);
    let want = "'a\\nb é\\nc\\n'\n8 ['A', 'B É', 'C']\n";
    assert_eq!((i.code, i.out.as_str()), (Some(0), want), "interpreter {}", i.err);
    assert_eq!(i, n, "native");
    if let Some(w) = w {
        assert_eq!((w.code, w.out.as_str()), (Some(0), want), "direct wasm {}", w.err);
    }
    let (i, n, _) = engines("nofile", "t = read_text(\"missing.txt\")\n");
    assert!(i.err.starts_with("error: No such file or directory: 'missing.txt'\n"), "{}", i.err);
    assert_eq!(i, n);
}

// ---- programs -----------------------------------------------------------------------

/// lang/viv/examples/xyz_atoms.viv with lib/xyz.viv: the output CPython gives
/// for the same parse (the XYZ text split, each field a float, the O-H distance).
#[test]
fn the_xyz_example() {
    let lib = std::fs::read_to_string(root().join("lib/xyz.viv")).unwrap();
    let src = std::fs::read_to_string(root().join("lang/viv/examples/xyz_atoms.viv")).unwrap();
    let (i, n, w) = engines_with("xyz", &src, &[("lib/xyz.viv", &lib)]);
    let want = "O  Z =  8    0.000000  0.000000  0.117300\nH  Z =  1    0.000000  0.757200 -0.469200\nH  Z =  1    0.000000 -0.757200 -0.469200\nr(OH) = 0.9578 A\n";
    assert_eq!((i.code, i.out.as_str(), i.err.as_str()), (Some(0), want, ""), "interpreter");
    assert_eq!(i, n, "native");
    if let Some(w) = w {
        assert_eq!((w.code, w.out.as_str()), (Some(0), want), "direct wasm {}", w.err);
    }
    // a bad line stops and says which
    let bad = "from xyz import parse_xyz\nm = parse_xyz(\"2\\nx\\nO 0 0 0\\nH 0 0\\n\")\n";
    let (i, n, _) = engines_with("xyzbad", bad, &[("lib/xyz.viv", &lib)]);
    assert!(i.err.starts_with("error: line 4: an atom is a symbol and three numbers, not 'H 0 0'\n"), "{}", i.err);
    assert_eq!(i, n, "native");
}

/// The string benchmarks (tools/strings/bench) at full size (10^6 characters,
/// tokens and numbers) print what CPython printed in every engine; the
/// timings are tools/strings/bench.sh's. The C and wasm runs are optimized
/// builds; the interpreter here is the test profile's.
#[test]
fn benchmarks_at_full_size() {
    for b in ["join", "split", "floats"] {
        let src = std::fs::read_to_string(root().join(format!("tools/strings/bench/{b}.viv"))).unwrap();
        let want = std::fs::read_to_string(root().join(format!("tools/strings/bench/{b}.out"))).unwrap();
        prints(b, &src, &want);
    }
}

/// `grad` differentiates a def that reads numbers from text (the text is a
/// constant to it), and refuses text made from the differentiated value.
#[test]
fn grad_and_strings() {
    prints(
        "grad",
        "COEF = \"2.5 1.5\"\ndef f(x) {\n    c = split(COEF)\n    return float(c[0]) * x * x + float(c[1]) * x\n}\nshow grad(f, 2.0)\n",
        "11.5\n",
    );
    refused("gradstr", "def f(x) {\n    s = str(x)\n    return x * x + len(s)\n}\nshow grad(f, 2.0)\n", "`grad` has no adjoint for `str`");
}

// ---- sorted and unique: Python's sorted(s), sorted(set(s)) ---------------------------

#[test]
fn sorted_and_unique_of_a_string_and_a_list() {
    // the expected lines are CPython's: sorted(T), sorted(set(T)), len(set(T)),
    // sorted(set(w)), sorted(set([])), sorted(""), "".join(sorted(set(T)))
    prints(
        "unique",
        "T = \"hello World! héllo\"\nprint(sorted(T))\nprint(unique(T))\nprint(len(unique(T)))\nw = split(\"b a c a B\")\nprint(unique(w))\nprint(unique([]))\nprint(sorted(\"\"))\nprint(join(\"\", unique(T)))\nprint(sorted([]))\n",
        "[' ', ' ', '!', 'W', 'd', 'e', 'h', 'h', 'l', 'l', 'l', 'l', 'l', 'o', 'o', 'o', 'r', 'é']\n[' ', '!', 'W', 'd', 'e', 'h', 'l', 'o', 'r', 'é']\n10\n['B', 'a', 'b', 'c']\n[]\n[]\n !Wdehloré\n[]\n",
    );
    // characters of 1 to 4 bytes sort by code point, which is their bytes' order
    prints("unique_wide", "print(unique(\"z𝕏éaé€z\"))\nprint(sorted(\"z𝕏éaé€z\"))\n", "['a', 'z', 'é', '€', '𝕏']\n['a', 'z', 'z', 'é', 'é', '€', '𝕏']\n");
    // a list's strings: equal strings once, prefixes first
    prints("unique_list", "print(unique([\"ab\", \"a\", \"\", \"ab\", \"b\", \"\"]))\n", "['', 'a', 'ab', 'b']\n");
    // the result is a list of strings like any other
    prints("unique_use", "v = unique(\"banana\")\nprint(len(v), v[0] + v[-1], \"n\" in v)\nfor c in v {\n    print(c)\n}\n", "3 an true\na\nb\nn\n");
}

#[test]
fn sort_and_unique_refuse_what_they_do_not_take() {
    refused("sort_list", "w = split(\"b a\")\nsort(w)\n", "`sort` sorts an array of numbers in place, and a list of strings is never changed in place");
    refused("sort_str", "s = \"ba\"\nsort(s)\n", "`sort` sorts an array of numbers in place, and a string is never changed in place");
}

/// `unique` and `sorted` of numbers are builtins now (lang/viv/std/builtins.viv),
/// in every engine.
#[test]
fn unique_and_sorted_of_numbers() {
    let (i, n, w) = engines("unique_num", "print(unique([3.0, 1.0, 3.0, -0.0, 0.0]), sorted([3.0, 1.0, 2.0]))\n");
    assert_eq!(i.out, "[0, 1, 3] [1, 2, 3]\n");
    assert_eq!(i, n);
    if let Some(w) = w {
        assert_eq!(w.out, i.out);
    }
}
