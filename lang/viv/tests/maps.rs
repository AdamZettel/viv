// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Maps: `map()`, `m[k] = v`, `m[k]`, `has`, `get`,
//! `remove`, `keys`, `values`, `copy`, `clear`, `len`, and printing.
//!
//! Each program runs in the interpreter (`viv run`), as C compiled natively,
//! and (when node and a viv.wasm with the map runtime are present) as a module
//! from the direct wasm backend. A program that runs must print the same bytes
//! in all three; one that stops must stop with the same words, and the
//! interpreter and C with the same location. (A wasm module reports the
//! message only, as it does for every run-time error.)


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
    let dir = std::env::temp_dir().join(format!("viv_maps_{}_{tag}", std::process::id()));
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
    let b = Command::new(cc()).current_dir(dir).args(["-std=c99", "-O2", "-Wall", "p.c", "-lm", "-o", "p"]).output().unwrap();
    assert!(b.status.success(), "the C did not build:\n{}", String::from_utf8_lossy(&b.stderr));
    Some(ran(Command::new(dir.join("p")).output().unwrap()))
}

/// The direct wasm backend in node, or None when there is no runtime to run it.
fn direct(dir: &Path) -> Option<Ran> {
    let wasm = std::env::var("VIV_WASM").map(PathBuf::from).unwrap_or_else(|_| root().join("www/public/viv/viv.wasm"));
    let bytes = std::fs::read(&wasm).ok()?;
    if !bytes.windows(11).any(|w| w == b"vrt_map_new") {
        eprintln!("skip wasm: {} has no map runtime (www/build-wasm.sh builds one)", wasm.display());
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
        assert_eq!(i, n, "interpreter and C disagree on\n{src}");
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

fn prints(tag: &str, src: &str, out: &str) {
    let r = all(tag, src);
    assert_eq!(r.code, Some(0), "{tag}: {r:?}");
    assert_eq!(r.out, out, "{tag}");
}

/// A program that stops: stderr is exactly `err`.
fn stops(tag: &str, src: &str, out: &str, err: &str) {
    let r = all(tag, src);
    assert_eq!(r.code, Some(4), "{tag}: {r:?}");
    assert_eq!(r.out, out, "{tag}");
    assert_eq!(r.err, err, "{tag}");
}

/// A program every engine refuses before it runs, with these words.
fn refused(tag: &str, src: &str, words: &str) {
    let dir = scratch(tag);
    std::fs::write(dir.join("p.viv"), src).unwrap();
    let r = interp(&dir);
    assert_eq!(r.code, Some(3), "{tag}: {r:?}");
    assert!(r.err.contains(words), "{tag}: {}", r.err);
    let c = Command::new(viv_bin()).current_dir(&dir).args(["p.viv", "-o", "p.c"]).output().unwrap();
    assert!(!c.status.success(), "{tag}: C accepted it");
    assert!(String::from_utf8_lossy(&c.stderr).contains(words), "{tag}: {}", String::from_utf8_lossy(&c.stderr));
}

#[test]
fn the_builtins() {
    prints(
        "basics",
        "m = map()\nm[3] = 1.5\nm[1] = 2\nm[3] = 4\nshow m\nprint(m)\nshow len(m)\nshow has(m, 1)\nshow has(m, 2)\nshow get(m, 7, -1)\nshow get(m, 1, -1)\n\
         show keys(m)\nshow values(m)\nx = remove(m, 1)\nshow x\nshow remove(m, 42, 0)\nshow m\nclear(m)\nshow m\nshow len(m)\nm[5] = 6\nshow m\n",
        "m: {3: 4, 1: 2}\n{3: 4, 1: 2}\n2\ntrue\nfalse\n-1\n2\n[3, 1]\n[4, 2]\nx: 2\n0\nm: {3: 4}\nm: {}\n0\nm: {5: 6}\n",
    );
}

#[test]
fn printing_in_formats() {
    prints(
        "formats",
        "m = map()\nm[1] = 2.5\nm[3] = 4\nm[-2] = 0.1\nprint(fmt(m, \"d\", \"\"))\nprint(fmt(m))\nprint(fmt(m, \".2f\"))\nprint(\"m =\", fmt(m, \"3d\", \"g\"), \"end\")\nprint(map())\nshow map()\n",
        "{1: 2.5, 3: 4.0, -2: 0.1}\n{1.0: 2.5, 3.0: 4.0, -2.0: 0.1}\n{1.00: 2.50, 3.00: 4.00, -2.00: 0.10}\nm = {  1: 2.5,   3: 4,  -2: 0.1} end\n{}\n{}\n",
    );
}

#[test]
fn insertion_order_through_updates_removals_and_rebuilds() {
    // an update keeps a key's place; a key removed and added again goes last;
    // enough keys to rebuild the table several times, with removals between
    prints(
        "order",
        "m = map()\nm[10] = 1\nm[20] = 2\nm[30] = 3\nm[10] = 11\nshow keys(m)\nx = remove(m, 20)\nm[20] = 22\nshow m\n\
         big = map()\nfor i to 1000 {\n    big[i * 7 % 1000] = i\n    if i % 3 == 0 {\n        x = remove(big, i * 7 % 1000)\n    }\n}\n\
         ks = keys(big)\nshow len(big)\nshow ks[0]\nshow ks[len(ks) - 1]\ns = 0\nfor i to len(ks) {\n    s = s + (i + 1) * ks[i] + big[ks[i]]\n}\nshow s\n",
        "[10, 20, 30]\nm: {10: 11, 30: 3, 20: 22}\n666\n7\n986\ns: 116679872\n",
    );
}

#[test]
fn keys_compare_as_numbers() {
    // 0.0 and -0.0 are one key, and the first inserted is kept, as CPython keeps
    // it; the infinities are keys; a NaN is never found
    prints(
        "zero",
        "m = map()\nm[-0.0] = 1\nm[0] = 2\nprint(fmt(m))\nshow len(m)\nm[1 / 0] = 3\nm[-1 / 0] = 4\nshow m\nshow m[1 / 0]\n\
         nan = 0 / 0\nshow has(m, nan)\nshow get(m, nan, 5)\nshow remove(m, nan, 6)\n\
         b = map()\nb[9007199254740992] = 1\nb[9007199254740993] = 2\nb[1e300] = 3\nb[5e-324] = 4\nprint(fmt(b, \"d\", \"d\"))\nshow len(b)\n",
        "{-0.0: 2.0}\n1\nm: {0: 2, inf: 3, -inf: 4}\n3\nfalse\n5\n6\n{9007199254740992: 2, 1000000000000000052504760255204420248704468581108159154915854115511802457988908195786371375080447864043704443832883878176942523235360430575644792184786706982848387200926575803737830233794788090059368953234970799945081119038967640880074652742780142494579258788820056842838115669472196386865459400540160: 3, 0: 4}\n3\n",
    );
}

#[test]
fn a_map_is_one_handle() {
    // a second name, a parameter, a record field and a closure all hold the
    // same map; `copy` makes another
    prints(
        "alias",
        "record Hist {\n    counts: Map\n    n: Num\n}\n\
         def add(q, k) {\n    q[k] = get(q, k, 0) + 1\n}\n\
         def count(xs) {\n    h = map()\n    for i to len(xs) {\n        add(h, xs[i])\n    }\n    return h\n}\n\
         m = count([5, 5, 2, 7, 5])\nb = m\nb[9] = 1\nshow m\nc = copy(m)\nc[4] = 4\nshow len(m)\nshow c\n\
         r = Hist { counts: m, n: 3 }\nr.counts[5] += 10\nshow m[5]\n\
         f = fn(x) -> get(m, x, -1)\nm[100] = 42\nshow f(100)\n\
         g = map()\ndef note(k) {\n    g[k] = 2 * k\n    return len(g)\n}\nshow note(3)\nshow note(4)\nshow g\n",
        "m: {5: 3, 2: 1, 7: 1, 9: 1}\n4\nc: {5: 3, 2: 1, 7: 1, 9: 1, 4: 4}\n13\n42\n1\n2\ng: {3: 6, 4: 8}\n",
    );
}

#[test]
fn an_indexed_parameter_learns_it_is_a_map() {
    // `add` is typed before `build` reaches it: indexing guesses an array, the
    // map that arrives revises it
    prints(
        "infer",
        "def add(q, k) {\n    q[k] = 1\n}\ndef build() {\n    m = map()\n    add(m, 3)\n    add(m, 1)\n    return m\n}\nshow build()\n",
        "{3: 1, 1: 1}\n",
    );
}

#[test]
fn a_map_in_a_loop_and_a_test() {
    prints(
        "loop",
        "for i to 3 {\n    t = map()\n    t[i] = i * i\n    show t\n}\nm = map()\nm[5] = 3\ntest \"maps in a test\" {\n    assert_true(has(m, 5))\n    assert_near(m[5], 3, 0)\n}\n",
        "t: {0: 0}\nt: {1: 1}\nt: {2: 4}\nPASS maps in a test\n",
    );
}

#[test]
fn run_time_errors_are_located_in_every_engine() {
    let cases = [
        ("read", "m = map()\nm[1] = 2\nshow m[3]\n", "error: the key 3 is not in the map\n  --> p.viv:3:6\n   3 | show m[3]\n            ^^^^ here\n"),
        ("frac", "m = map()\nx = m[2.5] + 1\n", "error: the key 2.5 is not in the map\n  --> p.viv:2:5\n   2 | x = m[2.5] + 1\n           ^^^^^^ here\n"),
        ("remove", "m = map()\nx = remove(m, -0.0)\n", "error: the key -0 is not in the map\n  --> p.viv:2:5\n   2 | x = remove(m, -0.0)\n           ^^^^^^^^^^^^^^^ here\n"),
        ("add", "m = map()\nm[7] += 1\n", "error: the key 7 is not in the map\n  --> p.viv:2:1\n   2 | m[7] += 1\n       ^^^^ here\n"),
        ("nan", "m = map()\nm[0 / 0] = 1\n", "error: a map key cannot be NaN\n  --> p.viv:2:1\n   2 | m[0 / 0] = 1\n       ^^^^^^^^^^^^ here\n"),
    ];
    for (tag, src, err) in cases {
        stops(tag, src, "", err);
    }
}

#[test]
fn refused_before_the_run() {
    refused("arr", "m = map()\na = [m, m]\n", "an array cannot hold maps");
    refused("value", "m = map()\nm[1] = [1, 2]\n", "a map value must be a number");
    refused("bool", "m = map()\nm[true] = 3\n", "a map key must be a number");
    refused("two", "m = map()\nm[1, 2] = 3\n", "a map takes one key");
    refused("read2", "m = map()\nshow m[1, 2]\n", "a map takes one key");
    refused("first", "a = [1, 2]\nshow has(a, 1)\n", "`has` takes a map first");
    refused("arity", "m = map()\nshow get(m, 1)\n", "`get` is written get(m, k, default)");
    refused("fmt", "x = 3\nprint(fmt(x, \"d\", \"d\"))\n", "`fmt` takes two formats only for a map");
    refused("eq", "m = map()\nn = map()\nshow m == n\n", "this is a map");
    refused("par", "m = map()\nparallel for i to 4 {\n    m[i] = i\n}\n", "a `parallel for` cannot use a map");
    refused("grad", "def f(x) {\n    m = map()\n    m[1] = x\n    return m[1] * x\n}\nshow grad(f, 3)\n", "`grad` differentiates numbers and real arrays; this is a map");
}

#[test]
fn a_def_named_like_a_map_builtin_is_its_own() {
    prints("own", "def get(a, b, c) {\n    return a + b + c\n}\nshow get(1, 2, 3)\n", "6\n");
}

// ------------------------------------------------ maps beside strings
//
// Strings and maps arrived on two branches. The builtins
// that take both kinds, or whose names sit next to each other, are resolved by
// the type of the first argument, the same way in every engine; each name is
// run with the kind it takes and refused, before the run, with the kind it
// does not.

/// Refused before the run by the interpreter, the C backend and the wasm
/// backend, each with these words.
fn refused_everywhere(tag: &str, src: &str, words: &str) {
    refused(tag, src, words);
    let dir = scratch(&format!("{tag}_w"));
    std::fs::write(dir.join("p.viv"), src).unwrap();
    let w = Command::new(viv_bin()).current_dir(&dir).args(["p.viv", "-o", "p.wasm"]).output().unwrap();
    assert!(!w.status.success(), "{tag}: the wasm backend accepted it");
    assert!(String::from_utf8_lossy(&w.stderr).contains(words), "{tag}: {}", String::from_utf8_lossy(&w.stderr));
}

#[test]
fn a_line_with_a_map_and_a_string() {
    // one print with both goes through viv_print_sm in C (tag 5 a map, 6 a string)
    prints(
        "mixed",
        "m = map()\nm[1] = 2.5\nm[3] = 4\ns = \"héllo\"\nw = split(\"a b\")\nprint(\"m =\", m, \"s =\", s, \"w =\", w)\n\
         print(fmt(m, \"d\", \"\"), \" \", fmt(s, \">7\"), \"|\", sep=\"\")\nprint(s, fmt(m, \".1f\"))\nshow m\nshow s\n",
        "m = {1: 2.5, 3: 4} s = héllo w = ['a', 'b']\n{1: 2.5, 3: 4.0}   héllo|\nhéllo {1.0: 2.5, 3.0: 4.0}\nm: {1: 2.5, 3: 4}\ns: 'héllo'\n",
    );
}

#[test]
fn len_of_a_string_a_list_of_strings_a_map_and_an_array() {
    prints(
        "len",
        "m = map()\nm[1] = 2\nm[5] = 6\nshow len(m)\nshow len(\"héllo\")\nshow len(split(\"a b c\"))\nshow len([1, 2, 3, 4])\nshow len(\"\")\nshow len(map())\n",
        "2\n5\n3\n4\n0\n0\n",
    );
}

#[test]
fn copy_of_a_map_and_an_array_not_of_a_string() {
    prints(
        "copy",
        "m = map()\nm[1] = 2\nc = copy(m)\nc[3] = 4\nshow len(m)\nshow len(c)\na = [1, 2]\nb = copy(a)\nb[0] = 9\nshow a[0]\n",
        "1\n2\n1\n",
    );
    refused_everywhere("copy_s", "s = \"ab\"\nt = copy(s)\n", "`copy` takes an array or a map");
    refused_everywhere("copy_l", "w = split(\"a b\")\nt = copy(w)\n", "`copy` takes an array or a map");
}

#[test]
fn count_and_index_of_strings_not_of_a_map() {
    prints(
        "count",
        "show count(\"abab\", \"ab\")\nshow count(split(\"a b a\"), \"a\")\nshow index(\"hello\", \"l\")\nshow index(split(\"x y z\"), \"z\")\n",
        "2\n2\n2\n2\n",
    );
    refused_everywhere("count_m", "m = map()\nm[1] = 1\nshow count(m, 1)\n", "`count` takes count(string, string");
    refused_everywhere("index_m", "m = map()\nm[1] = 1\nshow index(m, 1)\n", "`index` takes index(string, string");
}

#[test]
fn get_remove_has_keys_values_clear_of_a_map_not_of_a_string() {
    prints(
        "mapnames",
        "m = map()\nm[2] = 3\nm[4] = 5\nshow get(m, 2, 0)\nshow has(m, 4)\nshow keys(m)\nshow values(m)\nshow remove(m, 2)\nclear(m)\nshow len(m)\n",
        "3\ntrue\n[2, 4]\n[3, 5]\n3\n0\n",
    );
    for (name, call) in [
        ("get", "get(s, 1, 0)"),
        ("remove", "remove(s, 1)"),
        ("has", "has(s, 1)"),
        ("keys", "keys(s)"),
        ("values", "values(s)"),
        ("clear", "clear(s)"),
    ] {
        refused_everywhere(&format!("{name}_s"), &format!("s = \"ab\"\nx = {call}\n"), &format!("`{name}` takes a map first"));
        refused_everywhere(&format!("{name}_l"), &format!("s = split(\"a b\")\nx = {call}\n"), &format!("`{name}` takes a map first"));
    }
}

// ------------------------------------------------ string keys and values
//
// A map has one kind of key and one kind of value, each a number or a
// string, fixed by what the program stores. Its strings hash with the
// strings' FNV-1a (`viv_str_hash`, `strings::hash`) and compare by their
// bytes; it prints as Python prints a dict.

#[test]
fn string_keys_count_words_and_print_as_python() {
    prints(
        "words",
        "counts = map()\nfor w in split(\"the cat and the hat and the bat\") {\n    counts[w] = get(counts, w, 0) + 1\n}\nprint(counts)\nshow counts\nshow counts[\"the\"]\n\
         show has(counts, \"dog\")\nks = keys(counts)\nprint(join(\",\", ks))\nshow values(counts)\nshow len(counts)\ncounts[\"and\"] += 10\nshow counts[\"and\"]\n\
         print(fmt(counts, \"\", \"5.1f\"))\nprint(\"counts = \", counts, \" (\", len(ks), \" words)\", sep=\"\")\n",
        "{'the': 3, 'cat': 1, 'and': 2, 'hat': 1, 'bat': 1}\ncounts: {'the': 3, 'cat': 1, 'and': 2, 'hat': 1, 'bat': 1}\n3\nfalse\nthe,cat,and,hat,bat\n[3, 1, 2, 1, 1]\n5\n12\n\
         {'the':   3.0, 'cat':   1.0, 'and':  12.0, 'hat':   1.0, 'bat':   1.0}\ncounts = {'the': 3, 'cat': 1, 'and': 12, 'hat': 1, 'bat': 1} (5 words)\n",
    );
}

#[test]
fn string_values_under_number_and_string_keys() {
    prints(
        "values",
        "names = map()\nnames[8] = \"oxygen\"\nnames[1] = \"hydrogen\"\nprint(names)\ns = names[8]\nprint(upper(s))\nshow values(names)\nshow keys(names)\n\
         syms = map()\nsyms[\"O\"] = \"oxygen\"\nsyms[\"H\"] = \"hydrogen\"\nsyms[\"O\"] = \"oxygène\"\nshow syms\nprint(get(syms, \"C\", \"none\"))\nprint(get(syms, \"H\", \"none\"))\n\
         print(remove(syms, \"O\"))\nprint(remove(syms, \"N\", \"no N\"))\nshow len(syms)\nshow syms\nclear(syms)\nshow syms\nsyms[\"x\"] = \"it's\"\nshow syms\n",
        "{8: 'oxygen', 1: 'hydrogen'}\nOXYGEN\n['oxygen', 'hydrogen']\n[8, 1]\nsyms: {'O': 'oxygène', 'H': 'hydrogen'}\nnone\nhydrogen\noxygène\nno N\n1\n\
         syms: {'H': 'hydrogen'}\nsyms: {}\nsyms: {'x': \"it's\"}\n",
    );
}

#[test]
fn unicode_empty_and_near_keys() {
    // the empty string is a key; keys that differ in one byte, in one
    // accent, or only in length are different keys; a NUL is a character
    prints(
        "unicode",
        "m = map()\nm[\"\"] = 1\nm[\"é\"] = 2\nm[\"e\u{301}\"] = 3\nm[\"日本語\"] = 4\nm[\"a\"] = 5\nm[\"aa\"] = 6\nm[\"ab\"] = 7\nm[\"a\\0\"] = 8\nm[\"😀\"] = 9\n\
         print(m)\nshow m[\"\"]\nshow m[\"e\u{301}\"]\nshow m[\"日本語\"]\nshow has(m, \"日本\")\nshow has(m, \"a\\0\")\nfor k in keys(m) {\n    print(len(k), k)\n}\n",
        "{'': 1, 'é': 2, 'e\u{301}': 3, '日本語': 4, 'a': 5, 'aa': 6, 'ab': 7, 'a\\x00': 8, '😀': 9}\n1\n3\n4\nfalse\ntrue\n0 \n1 é\n2 e\u{301}\n3 日本語\n1 a\n2 aa\n2 ab\n2 a\u{0}\n1 😀\n",
    );
}

#[test]
fn many_string_keys_force_rebuilds_and_keep_insertion_order() {
    // 5000 keys through eleven rebuilds, a third removed, one put back:
    // CPython's dict gives 3334 keys, 'k7' first and 'k0' last, and the
    // weighted sum 18517749340
    prints(
        "many",
        "m = map()\nfor i to 5000 {\n    m[f\"k{mod(i * 7, 5000):d}\"] = i\n}\nfor i to 5000 {\n    if mod(i, 3) == 0 {\n        x = remove(m, f\"k{i:d}\")\n    }\n}\n\
         m[\"k0\"] = -1\nks = keys(m)\nshow len(m)\nprint(repr(ks[0]), repr(ks[-1]))\ns = 0\nfor j to len(ks) {\n    s = s + (j + 1) * m[ks[j]]\n}\nprint(fmt(s, \"d\"))\n",
        "3334\n'k7' 'k0'\n18517749340\n",
    );
}

#[test]
fn deletion_and_reinsertion_order() {
    // an update keeps a key's place; a key removed and added again goes last,
    // as in a Python dict
    prints(
        "reorder",
        "m = map()\nfor w in split(\"a b c d e\") {\n    m[w] = len(m)\n}\nm[\"b\"] = 10\nx = remove(m, \"a\")\nx = remove(m, \"c\")\nm[\"a\"] = 20\nm[\"f\"] = 30\nm[\"c\"] = 40\nprint(m)\n\
         for i to 100 {\n    m[\"tmp\"] = i\n    x = remove(m, \"tmp\")\n}\nprint(m)\nc = copy(m)\nc[\"z\"] = 1\nshow len(m)\nshow len(c)\n",
        "{'b': 10, 'd': 3, 'e': 4, 'a': 20, 'f': 30, 'c': 40}\n{'b': 10, 'd': 3, 'e': 4, 'a': 20, 'f': 30, 'c': 40}\n6\n7\n",
    );
}

#[test]
fn a_map_of_strings_through_defs_and_globals() {
    // a def stores strings in a map it is given, another returns a new one,
    // a third changes a module's map; each map is one handle
    prints(
        "defs",
        "def tally(m: Map, text) {\n    for w in split(text) {\n        m[w] = get(m, w, 0) + 1\n    }\n    return len(m)\n}\n\
         def labels(n) {\n    out = map()\n    for i to n {\n        out[i] = \"label \" + fmt(i, \"d\")\n    }\n    return out\n}\n\
         seen = map()\ndef note(w) {\n    seen[w] = upper(w)\n    return 0\n}\n\
         t = map()\nshow tally(t, \"x y x\")\nshow tally(t, \"y z\")\nprint(t)\nl = labels(3)\nprint(l)\nshow l[2]\nx = note(\"hi\")\nx = note(\"yo\")\nprint(seen)\n",
        "2\n3\n{'x': 2, 'y': 2, 'z': 1}\n{0: 'label 0', 1: 'label 1', 2: 'label 2'}\n'label 2'\n{'hi': 'HI', 'yo': 'YO'}\n",
    );
}

#[test]
fn a_missing_string_key_stops_in_its_repr() {
    stops(
        "missing_read",
        "m = map()\nm[\"a\"] = 1\nshow m[\"b\"]\n",
        "",
        "error: the key 'b' is not in the map\n  --> p.viv:3:6\n   3 | show m[\"b\"]\n            ^^^^^^ here\n",
    );
    stops(
        "missing_remove",
        "m = map()\nm[\"a\"] = \"x\"\nx = remove(m, \"it's\")\n",
        "",
        "error: the key \"it's\" is not in the map\n  --> p.viv:3:5\n   3 | x = remove(m, \"it's\")\n           ^^^^^^^^^^^^^^^^^ here\n",
    );
}

#[test]
fn a_map_mixing_kinds_is_refused_before_the_run() {
    // one kind of key and one kind of value per map, in every engine
    let note = "a map has one kind of";
    for (tag, src, words) in [
        ("skey", "m = map()\nm[\"a\"] = 1\nm[2] = 3\n", "this map's keys are strings, and this key is a number"),
        ("nkey", "m = map()\nm[2] = 3\nm[\"a\"] = 1\n", "this map's keys are strings, and this key is a number"),
        ("sval", "m = map()\nm[1] = \"a\"\nm[2] = 3\n", "this map's values are strings, and this value is a number"),
        ("sread", "m = map()\nm[1] = 2\nshow m[\"a\"]\nm[3] = 4\n", "this map's keys are strings, and this key is a number"),
        ("sdefault", "m = map()\nm[\"a\"] = 1\nshow get(m, \"b\", \"x\")\n", "this map's values are strings, and this value is a number"),
    ] {
        refused_everywhere(tag, src, words);
        refused_everywhere(tag, src, note);
    }
    refused_everywhere("slist", "m = map()\nm[1] = split(\"a b\")\n", "a map value must be a number or a string");
    refused_everywhere("bkey", "m = map()\nm[true] = 1\n", "a map key must be a number or a string");
    refused_everywhere("sfmt", "m = map()\nm[\"a\"] = 1\nprint(fmt(m, \"d\", \"\"))\n", "this map's keys are strings, which `fmt` writes as their repr");
}

/// The C of `src` built with AddressSanitizer prints what the interpreter
/// prints (None when the compiler has no AddressSanitizer).
fn asan_matches(tag: &str, src: &str) -> Option<()> {
    let dir = scratch(tag);
    std::fs::write(dir.join("p.viv"), src).unwrap();
    let c = Command::new(viv_bin()).current_dir(&dir).args(["p.viv", "-o", "p.c"]).output().unwrap();
    assert!(c.status.success(), "{tag}: {}", String::from_utf8_lossy(&c.stderr));
    let b = Command::new(cc()).current_dir(&dir).args(["-std=c99", "-O1", "-g", "-fsanitize=address", "p.c", "-lm", "-o", "p"]).output().ok()?;
    if !b.status.success() {
        eprintln!("skip {tag}: no AddressSanitizer build here");
        return None;
    }
    let r = ran(Command::new(dir.join("p")).output().unwrap());
    assert_eq!(r, interp(&dir), "{tag}: the AddressSanitizer build differs");
    Some(())
}

#[test]
fn maps_of_strings_run_clean_under_address_sanitizer() {
    // stores that replace and remove owned copies, reads that hand out new
    // strings, copies, clears and rebuilds, and a stop that names a key
    let src = "m = map()\nv = map()\nfor i to 3000 {\n    k = f\"key {mod(i * 13, 1000):d}\"\n    m[k] = get(m, k, 0) + i\n    v[mod(i, 50)] = k + \"!\"\n    if mod(i, 4) == 0 {\n        x = remove(m, k, -1)\n    }\n    if mod(i, 7) == 0 and has(v, 3) {\n        s = remove(v, 3)\n    }\n}\n\
               c = copy(v)\nclear(v)\nv[1] = \"one\"\nshow len(m)\nshow len(c)\nshow c[49]\nshow v\nshow keys(m)[:3]\nshow values(c)[:2]\n";
    asan_matches("asan_ok", src);
    asan_matches("asan_stop", "m = map()\nm[\"a\"] = \"x\"\nshow m[\"b\"]\n");
}

// ---- map comprehensions: Python's dict comprehensions -------------------------------

#[test]
fn map_comprehensions() {
    // the expected lines are CPython's for the same comprehensions
    prints(
        "comp_vocab",
        "TEXT = \"hello world! \" * 5\nvocab = unique(TEXT)\nstoi = {c: i for i, c in enumerate(vocab)}\nprint(stoi)\n\
         def encode(text: Str) {\n    return [stoi[c] for c in text]\n}\ndef decode(ids: Arr) {\n    return join(\"\", [vocab[i] for i in ids])\n}\n\
         ids = encode(\"hello\")\nprint(ids)\nprint(decode(ids))\n",
        "{' ': 0, '!': 1, 'd': 2, 'e': 3, 'h': 4, 'l': 5, 'o': 6, 'r': 7, 'w': 8}\n[4, 3, 5, 5, 6]\nhello\n",
    );
    // a filter, a key seen again (first place, last value), number and string kinds
    prints(
        "comp_kinds",
        "print({k: k * k for k in [1, 2, 3, 2] if k > 1})\nprint({w: len(w) for w in split(\"a bb a ccc\")})\nprint({i: c for i, c in enumerate(\"abca\")})\nprint({c: i for i, c in enumerate(\"abca\")})\n\
         print({i * 10 + j: i + j for i to 2 for j to 2})\nm = {x: 0 for x in []}\nprint(len(m))\n",
        "{2: 4, 3: 9}\n{'a': 1, 'bb': 2, 'ccc': 3}\n{0: 'a', 1: 'b', 2: 'c', 3: 'a'}\n{'a': 3, 'b': 1, 'c': 2}\n{0: 0, 1: 1, 10: 1, 11: 2}\n0\n",
    );
    // the loop variables are the comprehension's own, as in Python 3
    prints("comp_scope", "i = 7.0\nm = {i: 1 for i in [1, 2]}\nprint(i, m)\n", "7 {1: 1, 2: 1}\n");
    // a missing key stops the run where it is read
    stops("comp_missing", "stoi = {c: i for i, c in enumerate(\"ab\")}\nprint([stoi[c] for c in \"abc\"])\n", "", "error: the key 'c' is not in the map\n  --> p.viv:2:8\n   2 | print([stoi[c] for c in \"abc\"])\n              ^^^^^^^ here\n");
}

#[test]
fn a_brace_in_an_expression_is_only_a_map_comprehension() {
    refused("lit", "m = {1: 2}\n", "a brace in an expression is a map comprehension");
    refused("empty", "m = {}\n", "an empty map is `map()`");
    refused("set", "x = 1\nprint({x})\n", "Viv has no set");
    refused("setcomp", "s = {c for c in \"abc\"}\n", "Viv has no set");
}
