#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Random Viv programs that drive the hand-written C runtime where it is
riskiest: strings (slices with any bounds and steps, concatenation, search,
split and join over multi-byte text), maps (number and string keys and
values, removal, copies, clears, maps made in defs and returned or kept in
records), array slices (reads and writes, negative steps, rows and columns)
and the arena (loops that allocate, `while` conditions that allocate,
`continue` and `break` out of them, defs returning records of arrays,
strings and maps).

    tools/memcheck/fuzz/gen.py SEED > prog.viv

The interpreter is the oracle: tools/memcheck/fuzz/run.sh runs each program
there and as C built with AddressSanitizer and UndefinedBehaviorSanitizer,
and the two must print the same bytes (and stop with the same error, when a
random index is out of range). Every program is well typed; out-of-range
indices are allowed on purpose, and stop the run in both engines alike.
"""
import random
import sys

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
R = random.Random(seed)

TEXTS = ["", "a", "ab", "hello world", "  padded  ", "x,y,,z", "naïve café", "日本語テキスト", "é", "tab\tand\nnewline",
         "🙂 smile 🙂", "ABCdef123", "a-b-c-d-e-f", "    ", "ß and ﬃ", "1.5e3", "-42", "mixed 中文 and ascii"]
SEPS = [",", " ", "-", "a", "ab", "語", "🙂", ", "]
out = []


def lit(s):
    o = ['"']
    for c in s:
        if c == '"':
            o.append('\\"')
        elif c == "\\":
            o.append("\\\\")
        elif c == "\n":
            o.append("\\n")
        elif c == "\t":
            o.append("\\t")
        elif c == "{" or c == "}":
            o.append(c)
        else:
            o.append(c)
    o.append('"')
    return "".join(o)


def small(lo=-12, hi=12):
    return R.randint(lo, hi)


def sv():
    return f"s{R.randint(0, 3)}"


def slice_expr(base):
    parts = []
    for _ in range(R.choice([1, 2, 3])):
        parts.append(str(small()) if R.random() < 0.7 else "")
    if len(parts) == 3 and parts[2] in ("0", ""):
        parts[2] = R.choice(["1", "-1", "2", "-2", "3"])
    if len(parts) == 1:
        parts.append("")
    return f"{base}[{':'.join(parts)}]"


def arr_slice(base, n=12):
    """A slice of an array of n elements: steps of 1 or more and a start
    no later than the end, as arrays require (one in twenty breaks a rule, to
    check that both engines stop alike)."""
    if R.random() < 0.05:
        return slice_expr(base)
    lo = R.randint(0, n)
    hi = R.randint(lo, n)
    if R.random() < 0.3:
        lo, hi = lo - n, hi - n if hi < n else hi
    step = R.choice([1, 1, 1, 2, 3])
    parts = [str(lo) if R.random() < 0.8 else "", str(hi) if R.random() < 0.8 else ""]
    if step != 1:
        parts.append(str(step))
    if parts[0] and parts[1] and int(parts[0]) % (n + 1) > int(parts[1]) % (n + 1) and int(parts[1]) < 0 <= int(parts[0]):
        parts[1] = ""
    return f"{base}[{':'.join(parts)}]"


def string_stmt(d):
    a, b, c = sv(), sv(), sv()
    k = R.randint(0, 15)
    if k == 0:
        return [f"{a} = {b} + {c}"]
    if k == 1:
        return [f"{a} = {slice_expr(b)}"]
    if k == 2:
        return [f"{a} = {b} * {R.randint(0, 3)}"]
    if k == 3:
        return [f"{a} = upper({b}) + lower({c})"]
    if k == 4:
        return [f"{a} = replace({b}, {lit(R.choice(SEPS))}, {lit(R.choice(TEXTS[:6]))})"]
    if k == 5:
        return [f"print(find({b}, {lit(R.choice(SEPS))}), rfind({b}, {lit(R.choice(SEPS))}), count({b}, {lit(R.choice(SEPS))}))"]
    if k == 6:
        return [f"L = split({b}, {lit(R.choice(SEPS))})", f"print(len(L), {slice_expr('L')})", f"{a} = join({lit(R.choice(SEPS))}, L)"]
    if k == 7:
        return [f"{a} = strip({b}) + center({c}, {R.randint(0, 20)}, \"*\")"]
    if k == 8:
        return [f"print(repr({b}), len({b}), startswith({b}, {lit(R.choice(TEXTS[:5]))}))"]
    if k == 9:
        return [f"{a} = zfill({b}, {R.randint(0, 12)}) + ljust({c}, {R.randint(0, 9)}, \"-\")"]
    if k == 10:
        return [f"W = split({b})", f"W = append(W, {c})", "print(sorted(W))"]
    if k == 11:
        return [f"{a} = f\"<{{{b}}}|{{len({c})}}>\""]
    if k == 12:
        return [f"if len({b}) > 0 {{", f"    print({b}[{small(-6, 6)} % len({b})])", "}"]
    if k == 13:
        return [f"{a} = {lit(R.choice(TEXTS))}"]
    if k == 14:
        return [f"print(split({b}), splitlines({c}), rsplit({b}, \",\", {R.randint(-1, 3)}))"]
    return [f"print({a}, {b})"]


def map_stmt(d):
    k = R.randint(0, 11)
    key = R.randint(-5, 30)
    if k == 0:
        return [f"m[{key}] = {R.random() * 10:.3f}"]
    if k == 1:
        return [f"print(remove(m, {key}, -1.0), has(m, {key}), get(m, {key}, 0.0))"]
    if k == 2:
        return ["print(sum(keys(m)), sum(values(m)), len(m))"]
    if k == 3:
        return [f"sm[{lit('k' + str(key))}] = {sv()}"]
    if k == 4:
        return [f"print(get(sm, {lit('k' + str(key))}, \"none\"), len(sm))"]
    if k == 5:
        return ["m2 = copy(m)", f"m2[{key}] = 1.0", "print(len(m2), len(m))"]
    if k == 6:
        return ["print(keys(sm))", "print(values(sm))"]
    if k == 7:
        return [f"print(made({key}))"]
    if k == 8:
        return [f"bx = boxed({key}, {sv()})", "print(bx.m, bx.n, bx.t)"]
    if k == 9:
        return [f"if {R.randint(0, 9)} == 0 {{", "    clear(m)", "}"]
    if k == 10:
        return ["for kk to 3 {", f"    tmp = made(kk + {key})", "    acc = acc + len(tmp)", "}"]
    return ["print(m)"]


def array_stmt(d):
    k = R.randint(0, 9)
    if k == 0:
        return [f"print({arr_slice('a')})"]
    if k == 1:
        lo, hi = sorted([R.randint(0, 12), R.randint(0, 12)])
        return [f"a[{lo}:{hi}] = a[{lo}:{hi}] * 2.0 + 1.0"]
    if k == 2:
        return [f"print(M[{R.randint(-4, 3)}, :], M[:, {R.randint(-5, 4)}])"]
    if k == 3:
        return [f"M[{R.randint(0, 3)}, :] = M[{R.randint(0, 3)}, :] + 1.0"]
    if k == 4:
        return [f"a = concat(a[::{R.choice([1, 2, 3])}], a)[:12]"]
    if k == 5:
        return [f"print(sum(a), a[{small(-12, 11)}])"]
    if k == 6:
        return [f"a = roll(a, {R.randint(-3, 3)})"]
    if k == 7:
        return [f"print(transpose(M)[{R.randint(0, 4)}, {R.randint(0, 3)}])"]
    if k == 8:
        return ["r = pair(a)", "print(r.x[0], r.s, len(r.y))"]
    return ["print(shape(M), len(a))"]


def loop_stmt(d):
    body = []
    for _ in range(R.randint(1, 4)):
        body += any_stmt(d + 1)
    k = R.randint(0, 3)
    v = f"i{d}"
    if k == 0:
        head = [f"for {v} to {R.randint(0, 5)} {{"]
    elif k == 1:
        head = [f"n{d} = 0", f"while len(str(n{d}) + {sv()}) < {R.randint(1, 30)} and n{d} < 6 {{", f"    n{d} = n{d} + 1"]
    elif k == 2:
        head = [f"for {v} to {R.randint(1, 6)} {{", f"    if {v} % 2 == 1 {{", "        continue", "    }"]
    else:
        head = [f"for {v} to {R.randint(1, 6)} {{", f"    if {v} > {R.randint(0, 4)} {{", "        break", "    }"]
    return head + ["    " + l for l in body] + ["}"]


def any_stmt(d):
    x = R.random()
    if d < 2 and x < 0.15:
        return loop_stmt(d)
    if x < 0.5:
        return string_stmt(d)
    if x < 0.75:
        return map_stmt(d)
    return array_stmt(d)


out.append("""record Box {
    m: Map
    n: Num
    t: Str
}
record Pair {
    x: Arr
    y: Arr
    s: Str
}
def made(k) {
    mm = map()
    for i to 5 {
        mm[k + i] = i * 0.5
    }
    return mm
}
def boxed(k, t: Str) -> Box {
    q = map()
    q[k] = k * 2.0
    return Box { m: q, n: len(q), t: t + "!" }
}
def pair(v: Arr) -> Pair {
    w = v * 2.0
    return Pair { x: w + 1.0, y: v[1:], s: str(sum(w)) }
}""")
out.append("s0 = " + lit(R.choice(TEXTS)))
out.append("s1 = " + lit(R.choice(TEXTS)))
out.append("s2 = " + lit(R.choice(TEXTS)))
out.append("s3 = " + lit(R.choice(TEXTS)))
out.append("m = map()")
out.append("sm = map()")
out.append('sm["k0"] = "zero"')
out.append("acc = 0.0")
out.append("a = linspace(0.0, 1.0, 12)")
out.append("M = zeros(4, 5) + 1.5")
for _ in range(R.randint(10, 40)):
    out.extend(any_stmt(0))
out.append("print(s0, s1, s2, s3, len(m), len(sm), acc, sum(a), sum(M))")
print("\n".join(out))
