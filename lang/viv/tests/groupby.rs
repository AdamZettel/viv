// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Group-by on Int keys (lang/viv/std/groupby.viv): `factorize`,
//! `factorize2`, `match_rows` and the fused `group_sum`, `group_sum_num`, `group_sum2`, which a program gets by calling them. Each
//! program runs in the interpreter, as C compiled natively, and (with node
//! and a viv.wasm) from the direct wasm backend; all three must print the
//! same bytes.
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
    let dir = std::env::temp_dir().join(format!("viv_groupby_{}_{tag}", std::process::id()));
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
    if bytes.is_empty() {
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

#[test]
fn factorize_basics() {
    prints(
        "basics",
        "xs: [Int] = [5, -3, 5, 7, -3, 9223372036854775807, -9223372036854775807 - 1, 0, 7]\nu, ids = factorize(xs)\nshow u, ids\n\
         e: [Int] = []\nu0, i0 = factorize(e)\nshow len(u0), len(i0)\nsame: [Int] = [4, 4, 4]\nu1, i1 = factorize(same)\nshow u1, i1\n",
        "u: [5, -3, 7, 9223372036854775807, -9223372036854775808, 0]  ids: [0, 1, 0, 2, 1, 3, 4, 5, 2]\n0  0\nu1: [4]  i1: [0, 0, 0]\n",
    );
}

#[test]
fn factorize2_and_match_rows() {
    prints(
        "pairs",
        "a: [Int] = [1, 1, 2, 2, 1, -9223372036854775807 - 1]\nb: [Int] = [3, 4, 3, 3, 3, 9223372036854775807]\nka, kb, g = factorize2(a, b)\nshow ka, kb, g\n\
         bk: [Int] = [10, -4, 10, 7]\np: [Int] = [7, 8, 10, -4, -9223372036854775807 - 1]\nshow match_rows(bk, p)\n",
        "ka: [1, 1, 2, -9223372036854775808]  kb: [3, 4, 3, 9223372036854775807]  g: [0, 1, 2, 2, 0, 3]\n[3, -1, 0, 1, -1]\n",
    );
    let r = all("short", "a: [Int] = [1, 2]\nb: [Int] = [1]\nka, kb, g = factorize2(a, b)\n");
    assert_eq!(r.code, Some(4), "{r:?}");
    assert!(r.err.contains("factorize2: the key columns differ in length"), "{}", r.err);
}

/// Collisions: 3000 keys built to share low bits, high bits or sign, checked
/// against the definition (and against a linear search for distinctness),
/// and the extreme Ints.
#[test]
fn factorize_collisions_and_edges() {
    prints(
        "collide",
        "#| every row's key is its group's key, a new group is the next id, and no
#| two groups share a key (a linear search, for small inputs)
def bad_rows(keys: [Int], slow: Bool) -> Num {
    u, ids = factorize(keys)
    bad = 0
    ng: Int = 0
    for i to len(keys) {
        if u[ids[i]] != keys[i] {
            bad += 1
        }
        if ids[i] > ng {
            bad += 1
        }
        if ids[i] == ng {
            ng += 1
        }
    }
    if ng != len(u) {
        bad += 1
    }
    if slow {
        for a to len(u) {
            for b to a {
                if u[a] == u[b] {
                    bad += 1
                }
            }
        }
    }
    return bad
}

big: Int = 9223372036854775807
mixed: [Int] = []
x: Int = 12345
for i to 3000 {
    x = wrap_add(wrap_mul(x, 6364136223846793005), 1442695040888963407)
    k = shr(x, 52)
    if i % 3 == 0 {
        k = shl(k, 50)
    }
    if i % 3 == 1 {
        k = -k
    }
    mixed = append(mixed, k)
}
edges: [Int] = [big, -big - 1, 0, -1, 1, big, -big - 1, big - 1, -big, 0]
show bad_rows(mixed, true), bad_rows(edges, true)
u, ids = factorize(edges)
show u, ids
",
        "0  0\nu: [9223372036854775807, -9223372036854775808, 0, -1, 1, 9223372036854775806, -9223372036854775807]  ids: [0, 1, 2, 3, 4, 0, 1, 5, 6, 2]\n",
    );
}

/// 10^6 distinct keys (an odd multiplier is a bijection on 64 bits): the
/// table grows through 17 rehashes and every row is its own group.
#[test]
fn factorize_a_million_distinct_keys() {
    let src = "#| every row's key is its group's key, a new group is the next id, and no
#| two groups share a key (a linear search, for small inputs)
n: Int = 1000000
ks = int_zeros(n)
for i to n {
    ks[i] = wrap_mul(i, -7046029254386353131)
}
u2, ids2 = factorize(ks)
same = 0
for i to n {
    if ids2[i] == i and u2[i] == ks[i] {
        same += 1
    }
}
show len(u2), same
";
    prints("million", src, "1000000  same: 1000000\n");
}

/// The fused sums (direct path for keys spanning few values, hash path
/// otherwise, negative and extreme keys) against `factorize` and a loop over
/// its ids: same groups, same order, same sums and counts.
#[test]
fn fused_group_sums_match_factorize() {
    let src = r#"#| the fused sums against factorize and a loop over its ids; 1 when they differ
def check(keys: [Int], v: [Int], w: Arr) -> Num {
    u, ids = factorize(keys)
    s = int_zeros(len(u))
    t = zeros(len(u))
    c = int_zeros(len(u))
    for i to len(keys) {
        s[ids[i]] += v[i]
        t[ids[i]] += w[i]
        c[ids[i]] += 1
    }
    gu, gs, gc = group_sum(keys, v)
    hu, ht, hc = group_sum_num(keys, w)
    bad = 0
    if len(gu) != len(u) or len(hu) != len(u) {
        return 1
    }
    for g to len(u) {
        if gu[g] != u[g] or gs[g] != s[g] or gc[g] != c[g] or hu[g] != u[g] or ht[g] != t[g] or hc[g] != c[g] {
            bad = 1
        }
    }
    return bad
}

def check2(a: [Int], b: [Int], v: [Int]) -> Num {
    ka, kb, ids = factorize2(a, b)
    s = int_zeros(len(ka))
    for i to len(a) {
        s[ids[i]] += v[i]
    }
    xa, xb, xs, xc = group_sum2(a, b, v)
    if len(xa) != len(ka) {
        return 1
    }
    bad = 0
    for g to len(ka) {
        if xa[g] != ka[g] or xb[g] != kb[g] or xs[g] != s[g] {
            bad = 1
        }
    }
    return bad
}

n: Int = 5000
dense = int_zeros(n)
wide = int_zeros(n)
neg = int_zeros(n)
v = int_zeros(n)
w = zeros(n)
x: Int = 99
for i to n {
    x = wrap_add(wrap_mul(x, 6364136223846793005), 1442695040888963407)
    dense[i] = 3 + band(shr(x, 40), 63)
    wide[i] = shl(band(shr(x, 33), 15), 55)
    neg[i] = -7 - band(shr(x, 20), 1023)
    v[i] = band(shr(x, 10), 255) - 100
    w[i] = float(band(shr(x, 12), 4095)) / 7.0
}
show check(dense, v, w), check(wide, v, w), check(neg, v, w)
show check2(dense, neg, v), check2(wide, dense, v), check2(dense, dense, v)
u, ids = factorize(neg)
show len(u), u[0] == neg[0], ids[0], ids[n - 1] < len(u)
small: [Int] = [4, 2, 4, 9, 2]
sv: [Int] = [1, 2, 3, 4, 5]
a1, a2, a3 = group_sum(small, sv)
show a1, a2, a3
big: [Int] = [9223372036854775807, -9223372036854775807 - 1, 9223372036854775807]
bv: [Int] = [1, 2, 3]
b1, b2, b3 = group_sum(big, bv)
show b1, b2, b3
e: [Int] = []
c1, c2, c3 = group_sum(e, e)
show len(c1), len(c2), len(c3)
"#;
    prints(
        "fused",
        src,
        "0  0  0\n0  0  0\n1018  true  0  true\na1: [4, 2, 9]  a2: [4, 7, 4]  a3: [2, 2, 1]\nb1: [9223372036854775807, -9223372036854775808]  b2: [4, 2]  b3: [2, 1]\n0  0  0\n",
    );
    let r = all("fshort", "a: [Int] = [1, 2]\nb: [Int] = [1]\nu, s, c = group_sum(a, b)\n");
    assert_eq!(r.code, Some(4), "{r:?}");
    assert!(r.err.contains("group_sum: the key and value columns differ in length"), "{}", r.err);
}
