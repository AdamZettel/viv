# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Lines of code on the ported path, PySCF against lib/hf.

    python3 study/ports/pyscf/lines.py            writes results/lines.json, prints the table

"The ported path" is what ran, measured the same way on both sides:

- PySCF: every Python function in the pyscf package that executes while
  `gto.M(...)` builds the molecule and while `scf.RHF(mol).kernel()` (or
  UHF) runs, for each case of port.toml, found with sys.setprofile. The
  code lines of those functions are counted (the union of their line
  ranges, so a nested function is counted once).
- Viv: every def the timing program runs, from `viv graph <program> --run`
  (the interpreter's call graph), over the same cases; the code lines of
  those defs are counted.

Both are split into the molecule (atoms and basis set) and the SCF (the
integrals, the guess, the loop, DIIS, the eigensolver, the energy). The
counting rules are study/ports/harness/linecount.py's (no blanks, comments,
docstrings or tests; Viv's lone closing braces are counted, and reported
apart). Basis-set data (PySCF's .dat files, lib/basis's tables) is data,
not code, and is not counted.

C on the PySCF side, stated separately: libcvhf and the libcgto drivers
are PySCF's own C and their source is installed (pyscf/lib/vhf, gto); the
files on the path are counted with a C rule (no blanks, no comments).
libcint, which computes the integrals, is installed as a library only
(libcint.6.dylib): its source is not on this machine, so its lines are not
counted here.
"""
import ast
import json
import os
import re
import subprocess
import sys
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "study", "ports", "harness"))
import linecount  # noqa: E402

CONF = tomllib.load(open(os.path.join(HERE, "port.toml"), "rb"))
VIV = os.path.join(REPO, "target", "release", "viv")

MOLECULE_VIV = ("lib/hf/mole.viv", "lib/hf/molecule.viv", "lib/xyz.viv", "lib/basis/")
# what libcint does on PySCF's side: the integrals
INTEGRALS_VIV = ("lib/hf/boys.viv", "lib/hf/hermite.viv", "lib/hf/md_eri.viv", "lib/hf/md_onebody.viv",
                 "lib/hf/cart2sph.viv", "lib/hf/transform.viv", "lib/hf/basis.viv", "lib/hf/eri_store.viv")
# what libcvhf does: J and K from integrals (in core) or from the basis (direct)
JK_VIV = ("lib/hf/jk.viv", "lib/hf/jk_direct.viv", "lib/hf/md_direct.viv")
# basis-set tables: data written as defs that return arrays
DATA_DEF = re.compile(r"^basis_(sto3g|631g|631gs|631gss|ccpvdz|ccpvtz|def2svp|def2tzvp)(_|$)")
MOLECULE_PY = ("gto/mole.py", "gto/basis/", "gto/elements.py", "lib/parameters.py", "gto/ecp.py", "gto/pp_int.py", "data/")

PROBE = r'''
import json, os, sys
sys.path.insert(0, %(here)r)
import pyscf_side as ps
from pyscf import gto, scf, lib
import pyscf
root = os.path.dirname(pyscf.__file__)
seen = {"molecule": set(), "scf": set()}
phase = ["molecule"]
def prof(frame, event, arg):
    if event == "call":
        co = frame.f_code
        if co.co_filename.startswith(root):
            seen[phase[0]].add((os.path.relpath(co.co_filename, root), co.co_firstlineno, co.co_name))
sys.setprofile(prof)
for name in %(cases)r:
    c = ps.case_conf(name)
    phase[0] = "molecule"
    mol = ps.build(c)
    phase[0] = "scf"
    mf = ps.new_mf(mol, c, 1e-10)
    mf.chkfile = None
    mf.kernel()
sys.setprofile(None)
print(json.dumps({k: sorted(v) for k, v in seen.items()}))
print(root)
'''


def pyscf_functions():
    cases = [c["name"] for c in CONF["case"] if c["name"] in ("water_ccpvdz", "watercat_ccpvdz_uhf", "benzene_ccpvdz", "water_sto3g")]
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", PROBE % {"here": HERE, "cases": cases}],
                         capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
    lines = out.stdout.strip().splitlines()
    return json.loads(lines[0]), lines[1]


def py_function_lines(root, funcs):
    """Code lines of the functions, per file, as a union."""
    by_file = {}
    for fn, first, name in funcs:
        by_file.setdefault(fn, []).append((first, name))
    total = 0
    per_file = {}
    for fn, items in by_file.items():
        path = os.path.join(root, fn)
        text = open(path).read()
        code = linecount.py_code_lines(text)
        tree = ast.parse(text)
        spans = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                first = node.lineno
                if getattr(node, "decorator_list", None):
                    first = min([first] + [d.lineno for d in node.decorator_list])
                spans.setdefault(node.lineno, (first, node.end_lineno))
        keep = set()
        for first, name in items:
            if first in spans:
                a, b = spans[first]
                keep |= {l for l in code if a <= l <= b}
        per_file[fn] = len(keep)
        total += len(keep)
    return total, per_file


def viv_defs():
    """def name -> (file, first line, last line) for lib/ and the port's viv/."""
    out = {}
    for base in ("lib", "study/ports/pyscf/viv"):
        for root, _, files in os.walk(os.path.join(REPO, base)):
            for f in files:
                if not f.endswith(".viv"):
                    continue
                path = os.path.join(root, f)
                rel = os.path.relpath(path, REPO)
                src = open(path).read().split("\n")
                i = 0
                while i < len(src):
                    m = re.match(r"^def (\w+)\s*\(", src[i])
                    if m:
                        j = i
                        while j < len(src) and src[j] != "}":
                            j += 1
                        # the `#|` lines above are the def's intent, not code
                        out.setdefault(m.group(1), (rel, i + 1, j + 1))
                        i = j
                    i += 1
    return out


def viv_graph_nodes(prog):
    r = subprocess.run([VIV, "graph", prog, "--run"], cwd=REPO, capture_output=True, text=True)
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith("__graph__ "):
            return json.loads(line[len("__graph__ "):])["nodes"]
    raise RuntimeError(f"no call graph from {prog}: {r.stderr[:400]}")


def viv_lines(nodes, defs):
    groups = {"molecule": {}, "integrals": {}, "jk": {}, "scf": {}}
    for n in nodes:
        if n not in defs or DATA_DEF.match(n):
            continue
        rel, a, b = defs[n]
        if rel.startswith("study/ports/"):
            continue
        if rel.startswith(MOLECULE_VIV):
            g = "molecule"
        elif rel.startswith(INTEGRALS_VIV):
            g = "integrals"
        elif rel.startswith(JK_VIV):
            g = "jk"
        else:
            g = "scf"
        groups[g].setdefault(rel, []).append((a, b, n))
    res = {}
    for g, files in groups.items():
        tot = clo = 0
        per = {}
        for rel, spans in files.items():
            lines, closers = linecount.viv_code_lines(open(os.path.join(REPO, rel)).read())
            keep = {l for l in lines if any(a <= l <= b for a, b, _ in spans)}
            kc = {l for l in closers if l in keep}
            per[rel] = {"lines": len(keep), "closers": len(kc), "defs": sorted(n for _, _, n in spans)}
            tot += len(keep)
            clo += len(kc)
        res[g] = {"lines": tot, "closers": clo, "files": per, "defs": sum(len(v["defs"]) for v in per.values())}
    return res


def path_coverage(nodes, defs):
    """Intent coverage of the defs on the path (`viv lens coverage --json`):
    a def is explained when its module's lens has a `what` for it; stale and
    unstamped entries are counted apart."""
    files = sorted({defs[n][0] for n in nodes if n in defs and not defs[n][0].startswith("study/") and not DATA_DEF.match(n)})
    r = subprocess.run([VIV, "lens", "coverage", "--json"] + files, cwd=REPO, capture_output=True, text=True)
    d = json.loads(r.stdout)
    missing, stale, steps = set(), set(), [0, 0]
    for m in d["modules"]:
        for it in m["items"]:
            if it["kind"] == "what":
                missing.add(it["anchor"])
            elif it["kind"] in ("stale", "unstamped"):
                stale.add(it["anchor"])
        steps[0] += m["coverage"]["steps"][0]
        steps[1] += m["coverage"]["steps"][1]
    on = [n for n in nodes if n in defs and not defs[n][0].startswith("study/") and not DATA_DEF.match(n)]
    return {"defs": len(on), "explained": len([n for n in on if n not in missing]),
            "missing": sorted(n for n in on if n in missing), "stale_or_unstamped": sorted(n for n in on if n in stale),
            "steps_in_path_modules": steps}


def c_code_lines(path):
    text = open(path).read()
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    n = 0
    for line in text.split("\n"):
        s = line.split("//", 1)[0].strip()
        if s:
            n += 1
    return n


# the PySCF C on the path whose source is installed (pyscf/lib)
PYSCF_C = {
    "libcvhf (J/K)": ["vhf/nr_incore.c", "vhf/nr_direct.c", "vhf/nr_direct_dot.c", "vhf/optimizer.c", "vhf/fill_nr_s8.c"],
    "libcgto (integral drivers)": ["gto/fill_int2c.c", "gto/fill_int2e.c"],
    "libnp_helper": ["np_helper/pack_tril.c", "np_helper/np_helper.c"],
}


def main():
    funcs, root = pyscf_functions()
    py = {}
    for g in ("molecule", "scf"):
        tot, per = py_function_lines(root, [tuple(x) for x in funcs[g]])
        py[g] = {"lines": tot, "functions": len(funcs[g]), "files": per}
    defs = viv_defs()
    nodes = set()
    for prog in ("water_ccpvdz_t", "watercat_ccpvdz_uhf_t", "water_sto3g_t"):
        nodes |= set(viv_graph_nodes(os.path.join("study/ports/pyscf/viv", prog + ".viv")))
    # the direct path, on a molecule small enough for the interpreter's run
    direct = os.path.join(HERE, "results", "lines_direct.viv")
    os.makedirs(os.path.dirname(direct), exist_ok=True)
    src = open(os.path.join(HERE, "viv", "water_ccpvdz_t.viv")).read().replace("guess: GS_CORE,", "guess: GS_CORE, jk: INT_JK_DIRECT,")
    open(direct, "w").write(src)
    nodes |= set(viv_graph_nodes(os.path.relpath(direct, REPO)))
    viv = viv_lines(sorted(nodes), defs)
    cov = path_coverage(sorted(nodes), defs)
    c = {}
    lib = os.path.join(root, "lib")
    for g, files in PYSCF_C.items():
        c[g] = {f: c_code_lines(os.path.join(lib, f)) for f in files}
    out = {"pyscf_python": py, "viv": viv, "pyscf_c": c, "intent_coverage": cov,
           "pyscf_c_total": sum(sum(v.values()) for v in c.values()),
           "libcint": "not counted: installed as libcint.6.dylib only, no source on this machine"}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "lines.json"), "w"), indent=1, sort_keys=True)
    for g in ("integrals", "jk"):
        print("%-9s Viv %5d lines (%d lone braces) in %3d defs; PySCF does this in C (libcint, libcvhf)" % (g, viv[g]["lines"], viv[g]["closers"], viv[g]["defs"]))
    for g in ("molecule", "scf"):
        print("%-9s PySCF Python %5d lines in %3d functions | Viv %5d lines (%d of them lone braces) in %3d defs | ratio %.2f (%.2f without braces)" % (
            g, py[g]["lines"], py[g]["functions"], viv[g]["lines"], viv[g]["closers"], viv[g]["defs"],
            viv[g]["lines"] / max(1, py[g]["lines"]), (viv[g]["lines"] - viv[g]["closers"]) / max(1, py[g]["lines"])))
    print("intent on the path: %d of %d defs explained; missing %s; stale or unstamped %d (%s); steps explained %d of %d in the path's modules" % (
        cov["explained"], cov["defs"], ", ".join(cov["missing"]) or "none", len(cov["stale_or_unstamped"]), ", ".join(cov["stale_or_unstamped"]), cov["steps_in_path_modules"][0], cov["steps_in_path_modules"][1]))
    print("PySCF C on the path (source installed): %d lines; %s" % (out["pyscf_c_total"], out["libcint"]))
    for g, files in c.items():
        print("   %-28s %s" % (g, ", ".join("%s %d" % (k, v) for k, v in files.items())))


if __name__ == "__main__":
    main()
