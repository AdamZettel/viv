# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Run the claims in claims/ and record what each program computed.

A claim is a statement about a topic, specific and bounded, that a
program demonstrates. It is two files:

  claims/<topic>/<slug>.viv   a fenced `###` header (title, line, claim, topic,
                              section, reference, scope, lane, by, cost,
                              ...), then a Viv program that computes the
                              quantities the claim states and asserts
                              its bounds in test blocks
  claims/<topic>/<slug>.py    its vipyr twin (optional): the same
                              quantities through the vipyr API
                              (`import vipyr as vp`), the same bounds,
                              as PASS / FAIL lines (tools/claims/
                              claimlib.py)

A claim holds while both programs pass and agree: every value both
`show` under the same name must be equal within the header's `agree:`
relative tolerance (default 1e-12). Values whose names start with `time`
are wall-clock times and are not compared. Without a twin the claim is
"Viv only".

The header's `title:` is the claim in a few words and `line:` its key
numbers in one sentence; `claim:` is the full statement. Every number in
the three must be a literal of a program or round to a value one shows.
A recorded run is current while the programs are unchanged: the hash
covers the programs and `agree:`, not the header's words, so editing a
title or a scope does not need a rerun (--text checks the numbers).

    python3 tools/claims/run.py                  every fast claim, report
    python3 tools/claims/run.py --all            fast and heavy
    python3 tools/claims/run.py --heavy          only the heavy ones
    python3 tools/claims/run.py --only <slug>    the claims whose slug contains it
    python3 tools/claims/run.py --write          record into claims/results.json
    python3 tools/claims/run.py --check          exit 1 when a claim fails
    python3 tools/claims/run.py --list           the claims and their recorded status
    python3 tools/claims/run.py --list --json    the same as one JSON object, {"claims": [...]}
    python3 tools/claims/run.py --text           the numbers of each title, line and sentence
                                                 against the recorded runs (no run)

The .viv program is built natively (`viv build`, the exact build) and run
from the repository root; the .py twin runs under the interpreter its
`python:` header field names (default python3) with PYTHONPATH=python and
VIV set. Claims run one at a time.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLAIMS = ROOT / "claims"
RESULTS = CLAIMS / "results.json"
VIV = os.environ.get("VIV", str(ROOT / "target/release/viv"))
OUT = ROOT / "target/claims"
FIELD = re.compile(r"^([a-z][a-z_]*):[ \t]*(.*)$")
# the figure records a program prints (www/src/lib/postCheck.js's FIGURE_LINE): the
# page draws them, and they are kept apart from the output a reader reads
FIGURE_LINE = re.compile(r"^(figure|plot|plot2d|plot3d|scatter|quiver|bars|fill_between|rangebars|axes|curve|label|"
                         r"xtick|ytick|xscale|yscale|aspect|yright|grid|share|style|origin|colormap|clim|levels|"
                         r"circle|rect|segment|box|arrow|note|poly):")
FIGURE_BYTES = 60_000
SHOWN = re.compile(r"^([A-Za-z_][\w.]*(?:\[\d+\])?): (.*)$")


def header(src: str) -> dict:
    """The `key: value` fields of the first `###` fenced block."""
    lines = src.split("\n")
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "###")
    except StopIteration:
        return {}
    out, key = {}, None
    for ln in lines[start + 1:]:
        if ln.strip() == "###":
            break
        m = FIELD.match(ln)
        if m:
            key = m.group(1)
            out[key] = m.group(2).strip()
        elif key and ln.startswith(("  ", "\t")):
            out[key] += " " + ln.strip()   # a continuation line
        else:
            key = None
    return out


def program(src: str) -> str:
    """The source without its `###` header block: what a claim's hash covers.
    The header's words (claim, title, line, topic, scope, reference) can be
    edited without rerunning; the program, and the agreement tolerance the
    verdict depends on, cannot."""
    lines = src.split("\n")
    marks = [i for i, ln in enumerate(lines) if ln.strip() == "###"]
    if len(marks) < 2:
        return src
    return "\n".join(lines[:marks[0]] + lines[marks[1] + 1:])


def digest(src: str, twin: str | None) -> str:
    h = header(src)
    key = program(src) + "\nagree: " + h.get("agree", "") + "\n" + (twin or "")
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def discover() -> list[dict]:
    found = []
    for p in sorted(CLAIMS.rglob("*.viv")):
        if p.name.startswith("_"):
            continue
        src = p.read_text()
        h = header(src)
        if "claim" not in h:
            continue
        twin = p.with_suffix(".py")
        found.append({
            "slug": p.stem.replace("_", "-"), "path": str(p.relative_to(ROOT)), "header": h,
            "twin": str(twin.relative_to(ROOT)) if twin.exists() else None,
            "twin_header": header(twin.read_text()) if twin.exists() else {},
            "sha": digest(src, twin.read_text() if twin.exists() else None),
            "cost": h.get("cost", "fast"),
        })
    return found


def parse(out: str) -> tuple[list[dict], dict]:
    tests, values = [], {}
    for ln in out.splitlines():
        if ln.startswith("PASS "):
            tests.append({"name": ln[5:].strip(), "pass": True})
        elif ln.startswith("FAIL "):
            name, _, msg = ln[5:].partition(" -- ")
            tests.append({"name": name.strip(), "pass": False, "message": msg.strip()})
        else:
            m = SHOWN.match(ln)
            if m and len(values) < 400:
                values[m.group(1)] = m.group(2).strip()
    return tests, values


def run_program(path: Path, slug: str, h: dict, kind: str) -> dict:
    t0 = time.time()
    env = os.environ.copy()
    if kind == "viv":
        OUT.mkdir(parents=True, exist_ok=True)
        exe = OUT / slug
        b = subprocess.run([VIV, "build", str(path), "-o", str(exe)], cwd=ROOT, capture_output=True, text=True)
        if b.returncode != 0:
            return {"status": "fail", "error": (b.stderr or b.stdout).strip()[-800:], "tests": [], "values": {},
                    "seconds": round(time.time() - t0, 2), "output": ""}
        cmd = [str(exe)]
    else:
        py = os.path.expanduser(h.get("python", "python3"))
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "python"), str(ROOT / "tools/claims"),
                                             env.get("PYTHONPATH", "")])
        env["VIV"] = VIV
        cmd = [py, str(path)]
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env, timeout=3600)
        rc, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        rc, out, err = 124, "", "timed out after 3600 s"
    tests, values = parse(out)
    ok = rc == 0 and tests and all(t["pass"] for t in tests)
    # the engine's own records (__graph__, __steps__, __trace__) are not output a reader wants
    lines = [ln for ln in out.split("\n") if not ln.startswith("__")]
    figure = "\n".join(ln for ln in lines if FIGURE_LINE.match(ln))
    shown = "\n".join(ln for ln in lines if not FIGURE_LINE.match(ln))
    res = {"status": "pass" if ok else "fail", "tests": tests, "values": values,
           "seconds": round(time.time() - t0, 2), "output": shown[-6000:]}
    if figure:
        res["figure"] = figure[:FIGURE_BYTES]
    if not ok and not any(not t["pass"] for t in tests):
        res["error"] = (err or out).strip()[-800:] or "no test ran"
    return res


def num(v: str):
    try:
        return float(v)
    except ValueError:
        return None


def printed_half_unit(text: str) -> float:
    """Half a unit in the last digit a number was printed with."""
    t = text.strip().lower()
    mant, _, exp = t.partition("e")
    digits = mant.lstrip("-+")
    dec = len(digits.split(".")[1]) if "." in digits else 0
    if "." not in digits and len(digits.lstrip("0")) < 6:
        return 0.0          # a short integer is exact
    return 0.5 * 10.0 ** ((int(exp) if exp else 0) - dec)


def agreement(a: dict, b: dict, tol: float) -> list[dict]:
    """The values both programs show under one name, and whether they agree."""
    out = []
    for k in sorted(set(a) & set(b)):
        if k.startswith("time"):
            continue
        x, y = num(a[k]), num(b[k])
        if x is None or y is None:
            ok = a[k].strip() == b[k].strip()
        else:
            # a value is compared to the digits the shorter of the two printed
            # (Viv's `show` prints 6 significant digits)
            ok = abs(x - y) <= max(tol * max(1.0, abs(x), abs(y)), printed_half_unit(a[k]), printed_half_unit(b[k]))
        out.append({"name": k, "viv": a[k], "vipyr": b[k], "agree": ok})
    return out


STATED = re.compile(r"(?<![\w^/.\-])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?(?:e-?\d+)?)(%)?(?![\w/^])")
LITERAL = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?(?:e-?\d+)?)(?![\w])")


def plain_numbers(text: str) -> str:
    """A typeset sentence with its numbers in the plain form STATED reads.
    `$a \\times 10^{b}$` and `$10^{b}$` become `aeb` and `1eb`; `1{,}000` is
    1,000 and `5\\%` is 5%; `\\frac{a}{b}` is a/b (a ratio, not two numbers);
    the rest of an exponent or subscript (`x^{2}`, `\\log_{10}`, `T_c`) is a
    name, not a quantity, and is dropped, as `x^2` was in plain text."""
    t = text.replace("$", " ").replace("{,}", ",").replace("\\%", "%").replace("−", "-")
    t = re.sub(r"\\[,;:! ]", " ", t)
    t = re.sub(r"\\[td]?frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*\\(?:times|cdot)\s*10\^\{?\s*([-+]?\d+)\s*\}?",
               lambda m: f"{m.group(1)}e{int(m.group(2))}", t)
    t = re.sub(r"(?<![\w.^{])10\^\{?\s*([-+]?\d+)\s*\}?", lambda m: f"1e{int(m.group(1))}", t)
    t = re.sub(r"[\^_]\{[^{}]*\}|[\^_]\\?[\w+-]+", "^", t)   # 0.95^{20}: the base is not stated either
    t = re.sub(r"\\[A-Za-z]+", " ", t)
    return t


def stated_numbers(text: str) -> list[tuple[str, float, float, bool]]:
    """The numbers a claim's sentence states: (text, value, half a unit of
    its last digit, whether it is a percentage). A number inside a word
    (oxygen-16, SLy5, 1p3/2, alpha^2) is a name, not a quantity. The
    sentence may be typeset ($...$ math, see plain_numbers)."""
    out = []
    for m in STATED.finditer(plain_numbers(text)):
        t = m.group(1).replace(",", "")
        mant = t.split("e")[0]
        exp = int(t.split("e")[1]) if "e" in t else 0
        dec = len(mant.split(".")[1]) if "." in mant else 0
        out.append((m.group(0), float(t), 0.5 * 10.0 ** (exp - dec), bool(m.group(2))))
    return out


def unlinked(c: dict, values: list[dict]) -> list[str]:
    """The numbers of the claim's title, line and sentence that neither program establishes:
    each must be a literal of a program (a bound a test asserts, a setting)
    or round to a value a program shows."""
    srcs = program((ROOT / c["path"]).read_text()) + "\n" + (program((ROOT / c["twin"]).read_text()) if c["twin"] else "")
    code = "\n".join(ln for ln in srcs.split("\n") if not re.match(r"^\s*#", ln))
    lits = {float(x) for x in LITERAL.findall(code)}
    # every number a program shows, the elements of an array included
    shown = [float(t) for vs in values for v in vs.values()
             for t in re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", v, re.I)]
    shown += [abs(v) for v in shown]   # a sentence states a magnitude as "below -0.17" or "about 0.81 below zero"
    bad = []
    h = c["header"]
    said = [h.get(k, "") for k in ("title", "line", "claim")]
    for text, x, half, pct in [n for t in said for n in stated_numbers(t)]:
        cands = [x, x / 100.0] if pct else [x]
        if any(v in lits for v in cands):
            continue
        if any(abs(v - w) <= (half / 100.0 if pct and w == x / 100.0 else half) * (1 + 1e-9) for v in shown for w in cands):
            continue
        bad.append(text)
    return bad


def run_one(c: dict) -> dict:
    v = run_program(ROOT / c["path"], c["slug"], c["header"], "viv")
    res = {"viv": v}
    status = v["status"]
    if c["twin"]:
        p = run_program(ROOT / c["twin"], c["slug"], c["twin_header"], "py")
        res["vipyr"] = p
        tol = float(c["header"].get("agree", "1e-12"))
        res["agreement"] = agreement(v["values"], p["values"], tol)
        if p["status"] != "pass" or not all(x["agree"] for x in res["agreement"]):
            status = "fail"
    loose = unlinked(c, [v["values"]] + ([res["vipyr"]["values"]] if "vipyr" in res else []))
    if loose:
        res["unlinked"] = loose
        status = "fail"
    res["status"] = status
    res["seconds"] = round(v["seconds"] + res.get("vipyr", {}).get("seconds", 0), 2)
    return res


def summary(r: dict) -> str:
    bad = []
    for side in ("viv", "vipyr"):
        x = r.get(side)
        if not x:
            continue
        bad += [f"{side}: {t['name']}" for t in x["tests"] if not t["pass"]]
        if x.get("error"):
            bad.append(f"{side}: {x['error'][-300:]}")
    bad += [f"the sentence states {x}, which no test asserts and no program shows" for x in r.get("unlinked", [])]
    bad += [f"disagree {a['name']}: {a['viv']} vs {a['vipyr']}" for a in r.get("agreement", []) if not a["agree"]]
    return "; ".join(bad)


def main() -> int:
    a = sys.argv[1:]
    only = a[a.index("--only") + 1] if "--only" in a else None
    found = discover()
    prev = json.loads(RESULTS.read_text()) if RESULTS.exists() else {"claims": {}}
    if "--text" in a:
        # the numbers of each claim's title, line and sentence against its recorded
        # run, without running: the words can change without a rerun, but not
        # to a number the programs do not establish
        bad = 0
        for c in found:
            r = prev["claims"].get(c["slug"])
            if not r or r.get("sha") != c["sha"]:
                continue
            loose = unlinked(c, [r["viv"]["values"]] + ([r["vipyr"]["values"]] if r.get("vipyr") else []))
            for x in loose:
                print(f"FAIL {c['path']}: the claim states {x}, which no test asserts and no program shows")
                bad += 1
            if "title" not in c["header"] or "line" not in c["header"]:
                print(f"FAIL {c['path']}: the header needs a title: and a line:")
                bad += 1
        print(f"claim text: {len(found)} claims, {bad} problem{'' if bad == 1 else 's'}")
        return 1 if bad else 0
    if "--list" in a:
        rows = []
        for c in found:
            if only and only not in c["slug"]:
                continue
            r = prev["claims"].get(c["slug"], {})
            st = r.get("status", "unrun") if r.get("sha") == c["sha"] else ("stale" if r else "unrun")
            twin = "both" if c["twin"] else "viv"
            h = c["header"]
            if "--json" in a:
                rows.append({"slug": c["slug"], "status": st, "topic": h.get("topic"), "section": h.get("section"),
                             "title": h.get("title"), "line": h.get("line"), "claim": h.get("claim"),
                             "reference": h.get("reference"), "scope": h.get("scope"), "lane": h.get("lane"),
                             "cost": c["cost"], "viv": c["path"], "twin": c["twin"],
                             "ran": r.get("ran"), "values": (r.get("viv") or {}).get("values", {})})
            else:
                print(f"{st:6} {twin:4} {h.get('topic', h.get('track', '?')):8} {c['slug']:42} {h.get('title', h['claim'])[:80]}")
        if "--json" in a:
            print(json.dumps({"claims": rows}))
        return 0
    todo = found
    if only:
        todo = [c for c in found if only in c["slug"]]
    elif "--all" not in a:
        want = "heavy" if "--heavy" in a else "fast"
        todo = [c for c in found if c["cost"] == want]
    failed = []
    for c in todo:
        r = run_one(c)
        r.update({"sha": c["sha"], "path": c["path"], "twin": c["twin"], "ran": time.strftime("%Y-%m-%d")})
        prev["claims"][c["slug"]] = r
        print(f"{r['status']:5} {r['seconds']:7.2f}s  {'both' if c['twin'] else 'viv '}  {c['slug']}"
              + (f"  -- {summary(r)}" if r["status"] != "pass" else ""), flush=True)
        if r["status"] != "pass":
            failed.append(c["slug"])
    known = {c["slug"] for c in found}
    prev["claims"] = {k: v for k, v in sorted(prev["claims"].items()) if k in known}
    if "--write" in a:
        prev["about"] = "What each claim's programs computed when last run: tools/claims/run.py --write."
        RESULTS.write_text(json.dumps(prev, indent=1) + "\n")
    print(f"claims: {len(todo) - len(failed)} of {len(todo)} pass")
    return 1 if failed and "--check" in a else 0


if __name__ == "__main__":
    sys.exit(main())
