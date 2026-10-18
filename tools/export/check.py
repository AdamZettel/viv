# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The exact-bits check of an exported program.

    python3 tools/export/check.py dft/pwdft.export.toml [--decks si,ge] [--no-interp]
                                  [--cc CC] [--record] [--out DIR]
    python3 tools/export/check.py study/ports/pyscf/vivscf.export.toml [--cases a,b] ...

For each deck (or, for a program without decks whose manifest names
`cases`, each case: a run with the parameters the cases file gives it) it
runs the program three ways and requires the same bits:

  library      `viv export`, the exact build of the C file, called through the
               generated Python wrapper;
  native       the same Viv files compiled as an ordinary program
               (`viv --use deck --use files... --use checks report.viv`, the
               report printing every result in full) and built with
               `cc -O2 -ffp-contract=off`;
  interpreter  `viv run` of the same files (skipped with --no-interp).

For a case, the native and interpreted runs read a copy of each program file
with every given parameter's assignment line rewritten to the value
(`with_params`), under target/export/check_tmp: the native build has no
parameter tables, so the value is written into the source, as
`viv run --set` does for numbers.

"The same" means: the library's printed lines equal the native program's
stdout and the interpreter's (less the report's lines and the interpreter's
__graph__/__trace__ records), byte for byte; and the SHA-256 of the results
(test_<name>.py's result_hash: every result's bytes) is the same from all
three. Every test of the check file passes in the library run.

--cc builds the library with another compiler as well (gcc-15, say) and adds
its hash; with --record the hashes go to <name>.exact_hashes.json beside the
manifest, which `viv export` copies next to the program for its own test.
Prints a table: deck, seconds per engine, hash, agreement.
"""
import argparse
import hashlib
import importlib
import json
import os
import struct
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIV = os.environ.get("VIV", os.path.join(ROOT, "target", "release", "viv"))


def result_hash(r):
    """test_<name>.py's result_hash: SHA-256 of each result's name and bytes."""
    h = hashlib.sha256()
    for name in sorted(k for k in r if k not in ("tests", "log")):
        v = r[name]
        h.update(name.encode() + b"\0")
        if v is None:
            h.update(b"none")
        elif isinstance(v, bool):
            h.update(struct.pack("<d", float(v)))
        elif isinstance(v, int):
            h.update(struct.pack("<q", v))
        elif isinstance(v, float):
            h.update(struct.pack("<d", v))
        elif isinstance(v, str):
            h.update(v.encode())
        else:
            h.update(struct.pack("<q", v.ndim) + b"".join(struct.pack("<q", d) for d in v.shape))
            h.update(v.astype("<f8" if v.dtype.kind == "f" else "<i8").tobytes())
    return h.hexdigest()


def viv_literal(spec, v):
    """The Viv text of value `v` of a parameter of the manifest's `spec`."""
    kind = spec if isinstance(spec, str) else spec["type"]

    def num(x):
        x = float(x)
        r = repr(x)
        return r if any(c in r for c in ".en") else r + ".0"

    if isinstance(spec, dict) and "labels" in spec:
        rule = spec.get("normalize")

        def norm(s):
            low = s.lower()
            if rule == "pyscf_basis":
                return "".join("s" if c == "*" else c for c in low if c not in "-_ ")
            return low if rule else s
        for label, variant in spec["labels"].items():
            if norm(label) == norm(v):
                return "%s.%s" % (kind, variant)
        raise ValueError("%r is not one of %s" % (v, ", ".join(spec["labels"])))
    if kind == "Num":
        return num(v)
    if kind == "Int":
        return str(int(v))
    if kind == "Bool":
        return "true" if v else "false"
    if kind in ("Str", "Enum"):
        return json.dumps(v)
    if kind in ("IntArr", "[Int]"):
        return "[" + ", ".join(str(int(x)) for x in v) + "]"
    if kind == "Arr":
        def arr(x):
            return "[" + ", ".join(arr(y) for y in x) + "]" if isinstance(x, (list, tuple)) else num(x)
        return arr(v)
    raise ValueError("no Viv literal for a %s" % kind)


def with_params(m, params, tmp):
    """The manifest's files, copied under `tmp` with each parameter in
    `params` written into its assignment line (one top-level line per
    parameter, `NAME = ...` or `NAME: T = ...`, its value on that line)."""
    import re
    texts = {f: open(os.path.join(ROOT, f)).read().split("\n") for f in m["files"]}
    for key, v in params.items():
        # a case names a parameter as the wrapper does, in lower case; the
        # manifest (and the program's line) may spell it in upper case
        name = next((k for k in m["params"] if k.lower() == key.lower()), key)
        pat = re.compile(r"^(" + re.escape(name) + r"(?:\s*:\s*[^=]+?)?)\s*=\s*(.*)$")
        hits = [(f, i) for f, ls in texts.items() for i, l in enumerate(ls) if pat.match(l)]
        if len(hits) != 1:
            raise SystemExit("check.py: %d top-level lines assign %s; a case needs one" % (len(hits), name))
        f, i = hits[0]
        rhs = pat.match(texts[f][i]).group(2).split("#")[0]
        if rhs.count("[") != rhs.count("]") or rhs.count("(") != rhs.count(")"):
            raise SystemExit("check.py: %s:%d: the value of %s goes on past its line" % (f, i + 1, name))
        texts[f][i] = "%s = %s" % (pat.match(texts[f][i]).group(1), viv_literal(m["params"][name], v))
    out = []
    for f, ls in texts.items():
        path = os.path.join(tmp, f)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write("\n".join(ls))
        out.append(os.path.relpath(path, ROOT))
    return out


def parse_report(text, results):
    """The program's own lines, and the results its report printed."""
    import numpy as np
    lines, vals, shapes = [], {}, {}
    for line in text.splitlines():
        if line.startswith("SHAPE "):
            _, name, rest = line.split(" ", 2)
            shapes[name] = [int(x) for x in rest.strip("[]").split(",") if x.strip()]
        elif line.startswith("RESULT "):
            _, name, k, v = line.split(" ", 3)
            vals.setdefault(name, []).append((int(k), v))
        elif not line.startswith(("__graph__ ", "__trace__ ")):
            lines.append(line)
    out = {}
    for name, kind in results:
        key = name if name in vals else name.upper()
        if key not in vals:
            out[name] = None
        elif kind in ("Arr", "IntArr"):
            flat = [v for _, v in sorted(vals[key])]
            dt = np.float64 if kind == "Arr" else np.int64
            out[name] = np.array([float(x) if kind == "Arr" else int(x) for x in flat], dtype=dt).reshape(shapes[key])
        else:
            v = vals[key][0][1]
            out[name] = (int(float(v)) if kind == "Int" else bool(float(v)) if kind == "Bool"
                         else v if kind == "Str" else float(v))
    return lines, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest")
    ap.add_argument("--decks", default="")
    ap.add_argument("--cases", default="", help="for a manifest with cases: run only these")
    ap.add_argument("--no-interp", action="store_true")
    ap.add_argument("--cc", default="")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    manifest = os.path.relpath(os.path.abspath(a.manifest), ROOT)
    name = None
    for line in open(os.path.join(ROOT, manifest)):
        if line.startswith("name"):
            name = line.split("=", 1)[1].split("#")[0].strip().strip('"')
    out = a.out or os.path.join(ROOT, "target", "export", name)
    reports = out + "_reports"
    subprocess.run([VIV, "export", manifest, "-o", out, "--reports", reports], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    for f in os.listdir(out):
        if f.startswith("lib" + name):
            os.remove(os.path.join(out, f))
    sys.path.insert(0, out)
    t = time.time()
    mod = importlib.import_module(name)
    print(f"library built in {time.time() - t:.1f} s ({mod.version()})")
    # the manifest's decks and files, read back from the generated wrapper and the manifest
    try:
        import tomllib                # Python 3.11 and newer
    except ModuleNotFoundError:       # 3.9 and 3.10: the backport, `pip install tomli`
        import tomli as tomllib
    m = tomllib.load(open(os.path.join(ROOT, manifest), "rb"))
    decks = [d for d in m.get("decks", {})] or [""]
    if a.decks:
        decks = [d for d in decks if d in a.decks.split(",")]
    # a program without decks: its cases, each a set of parameters
    cases = {}
    if not m.get("decks") and m.get("cases"):
        # a case's parameters are its keys less `_expect` (values from a
        # reference outside Viv, which the generated test compares)
        cases = {c: {k: v for k, v in ps.items() if not k.startswith("_")}
                 for c, ps in json.load(open(os.path.join(ROOT, m["cases"]))).items()}
        decks = [c for c in cases if not a.cases or c in a.cases.split(",")]
    kwargs = (lambda d: cases[d]) if cases else (lambda d: {m.get("deck_param", "material"): d} if d else {})
    other = None
    if a.cc:
        lib = os.path.join(out, "lib%s_%s.so" % (name, os.path.basename(a.cc)))
        mod.build(lib, cc=a.cc)
        code = ("import sys, json; sys.path.insert(0, %r); import os; os.environ[%r] = %r; "
                "sys.path.insert(0, %r); import check; import %s as m; "
                "print(json.dumps({d: check.result_hash(m.run(**k)) for d, k in %r}))"
                % (out, name.upper() + "_LIB", lib, os.path.dirname(os.path.abspath(__file__)), name, [(d, kwargs(d)) for d in decks]))
        other = (os.path.basename(a.cc), code)
    rows, hashes, ok = [], {}, True
    tmp = os.path.join(ROOT, "target", "export", "check_tmp")
    os.makedirs(tmp, exist_ok=True)
    for d in decks:
        spec = m["decks"][d] if d and not cases else {}
        if cases:
            files = with_params(m, cases[d], os.path.join(tmp, name + "_" + d + "_src"))
            report = os.path.join(reports, "report.viv")
        else:
            files = ([spec["deck"]] if d else []) + m["files"] + ([spec["checks"]] if spec.get("checks") else [])
            report = os.path.join(reports, (d + "_" if d else "") + "report.viv")
        args = sum((["--use", f] for f in files), []) + [os.path.relpath(report, ROOT)]
        t = time.time()
        r = mod.run(**kwargs(d))
        t_lib = time.time() - t
        h_lib = result_hash(r)
        failed = [x for x in r["tests"] if not x[1]]
        c = os.path.join(tmp, f"{name}_{d or 'run'}.c")
        subprocess.run([VIV] + args + ["-o", c], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["cc", "-O2", "-ffp-contract=off", "-w", c, "-o", c[:-2], "-lm"], check=True)
        t = time.time()
        nat = subprocess.run([c[:-2]], capture_output=True, text=True)
        t_nat = time.time() - t
        nat_lines, nat_res = parse_report(nat.stdout, mod.RESULTS)
        h_nat = result_hash(dict(nat_res, tests=None, log=None))
        same_nat = nat_lines == r["log"] and h_nat == h_lib
        if not same_nat:
            stem = os.path.join(tmp, f"{name}_{d or 'run'}")
            open(stem + ".native.txt", "w").write(nat.stdout + nat.stderr)
            open(stem + ".library.txt", "w").write("\n".join(r["log"]) + "\n")
            print("  native: lines %s, results %s (status %d); outputs in %s.{native,library}.txt"
                  % ("same" if nat_lines == r["log"] else "DIFFERENT", "same" if h_nat == h_lib else "DIFFERENT", nat.returncode, os.path.relpath(stem, ROOT)))
        t_int, h_int, same_int = None, None, None
        if not a.no_interp:
            t = time.time()
            itp = subprocess.run([VIV, "run"] + args, cwd=ROOT, capture_output=True, text=True)
            t_int = time.time() - t
            int_lines, int_res = parse_report(itp.stdout, mod.RESULTS)
            h_int = result_hash(dict(int_res, tests=None, log=None))
            same_int = int_lines == r["log"] and h_int == h_lib
            # the interpreter stops a run past its step budget (interp.rs,
            # STEP_BUDGET, 2e9): a limit of the engine, not a disagreement
            over = itp.returncode != 0 and "step budget exceeded" in itp.stdout + itp.stderr
            if over:
                same_int, h_int = None, None
                print("  interpreter: stopped at its step budget after %.0f s; library and native only" % t_int)
            elif not same_int:
                # what differs, for the bug report: the lines or the results
                stem = os.path.join(tmp, f"{name}_{d or 'run'}")
                open(stem + ".interp.txt", "w").write(itp.stdout + itp.stderr)
                open(stem + ".library.txt", "w").write("\n".join(r["log"]) + "\n")
                print("  interpreter: lines %s, results %s (status %d); outputs in %s.{interp,library}.txt"
                      % ("same" if int_lines == r["log"] else "DIFFERENT", "same" if h_int == h_lib else "DIFFERENT", itp.returncode, os.path.relpath(stem, ROOT)))
        good = same_nat and same_int is not False and not failed and r["tests"]
        ok &= bool(good)
        hashes[d] = h_lib
        rows.append((d or "-", len(r["tests"]) - len(failed), len(r["tests"]), t_lib, t_nat, t_int, h_lib[:16], same_nat, same_int))
        print("%-20s tests %2d/%-2d  library %7.2f s  native %7.2f s  interpreter %s  sha256 %s  native %s  interpreter %s"
              % (d or "-", len(r["tests"]) - len(failed), len(r["tests"]), t_lib, t_nat,
                 "%7.1f s" % t_int if t_int is not None else "   -   ", h_lib[:16],
                 "same" if same_nat else "DIFFERENT", ("over budget" if t_int is not None else "-") if same_int is None else "same" if same_int else "DIFFERENT"), flush=True)
    if other:
        res = subprocess.run([sys.executable, "-c", other[1]], capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stderr)
            ok = False
        else:
            got = json.loads(res.stdout.strip().splitlines()[-1])
            for d in decks:
                same = got[d] == hashes[d]
                ok &= same
                print("%-20s %s build: sha256 %s  %s" % (d or "-", other[0], got[d][:16], "same" if same else "DIFFERENT"))
    if a.record and ok:
        # merged into the file: a run of some decks or cases updates theirs
        path = os.path.join(ROOT, os.path.dirname(manifest), name + ".exact_hashes.json")
        if os.path.exists(path):
            hashes = dict(json.load(open(path)), **hashes)
        with open(path, "w") as f:
            json.dump(hashes, f, indent=2, sort_keys=True)
            f.write("\n")
        print("wrote", os.path.relpath(path, ROOT))
    print("ALL SAME" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
