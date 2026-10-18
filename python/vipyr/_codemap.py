# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The code maps of the vipyr functions, written by `python -m vipyr.docs`.

For each registered function this finds the Viv program behind it and asks
the compiler for its map (`viv graph <file> --map [--entry <def>]`,
lang/viv/src/codemap.rs): every def the entry reaches, the module each is
written in, its steps, loops and call sites, and the builtins it calls.

    www/public/codemap/<key>.json   one map per program and entry, with the
                                    source of every module it shows
    www/src/data/codemap.json       the index: function -> map, the counts the
                                    pages quote, and the gallery entries that
                                    call each function

Where the program is, per kind of registry entry:

    program   the last file of the manifest's `files` (the program; the
              others are modules it uses), from its top level
    kernel    the adapter def the manifest's `viv = "file:def"` names
    inferred, feature   the registry's `source` and `entry`
    python    the registry's `map = "file:def"`: the Viv library behind the
              Python driver (vp.ode: lib/ode/solve.viv)

EXTRA lists the vipyr modules outside the registry that run a Viv program.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from . import _core, _registry, _spec

# vipyr modules with no registry entry, by the name the page uses
EXTRA = {
    "sisso.fit": dict(map="study/ports/sisso/sisso.viv", title="vipyr.sisso.fit",
                      about="SISSO symbolic regression (python/vipyr/sisso.py), the Viv port of SISSO 3.5"),
    "eigen.lowest_eigenpairs": dict(map="lib/eigen/lowest.viv:lowest_eigenpairs", title="lowest_eigenpairs",
                                    about="the contract lowest_eigenpairs (lib/eigen/lowest.viv): Lanczos or dense, "
                                          "used by vp.nuclear.shell_model"),
    "cc.run_c": dict(map="lib/cc/driver.viv:run_c", title="C compiler in Viv",
                     about="lib/cc and lib/rv32: C source to RISC-V machine code, run on the RV32IM simulator"),
}

FORMAT = 1


def site_index() -> Path:
    return _spec.repo_root() / "www" / "src" / "data" / "codemap.json"


def map_dir() -> Path:
    return _spec.repo_root() / "www" / "public" / "codemap"


def target_of(qual, reg, root: Path):
    """(file, entry def or None) of a registry entry, or None when it has no
    Viv program."""
    if reg.get("map"):
        f, _, e = reg["map"].partition(":")
        return f, e or None
    kind = reg["kind"]
    if kind == "program":
        man = _spec.read_toml(root / reg["manifest"])
        return man["files"][-1], None
    if kind == "kernel":
        man = _spec.read_toml(root / reg["manifest"])
        f, _, e = man["kernels"][reg["entry"]]["viv"].partition(":")
        return f, e
    if kind in ("inferred", "feature"):
        return reg["source"], reg["entry"]
    return None


def key_of(file, entry):
    k = file[:-4] if file.endswith(".viv") else file
    k = k.replace("/", ".")
    return k + ("--" + entry if entry else "")


def compile_map(root: Path, file, entry):
    cmd = [_core.compiler(), "graph", file, "--map"] + (["--entry", entry] if entry else [])
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=root, env=_core._env())
    if r.returncode != 0:
        raise RuntimeError("viv graph --map %s%s failed: %s" % (file, ":" + entry if entry else "", r.stderr.strip()[-500:]))
    m = json.loads(r.stdout)
    base = str(root.resolve()) + "/"
    for mod in m["modules"]:
        p = mod["path"]
        if p.startswith(base):
            p = p[len(base):]
        elif p.startswith(str(root) + "/"):
            p = p[len(str(root)) + 1:]
        if p.startswith("/") or p.startswith(".."):
            raise RuntimeError("viv graph --map %s: module %s is outside the repository (%s)" % (file, mod["name"], p))
        mod["path"] = p.removeprefix("./")
    add_sweeps(root, m)
    return m


def add_sweeps(root: Path, m):
    """Each contract's measurements: `<contract>_sweep.json`
    beside the program or the module that declares the contract, written by
    its sweep script, trimmed to what the map's pane shows."""
    for s in m.get("contracts", []):
        # beside the program first (a shared contract is swept per program),
        # then beside the module that declares the contract
        name = s["name"] + "_sweep.json"
        f = next((root / Path(m["modules"][k]["path"]).parent / name for k in (0, s["module"])
                  if (root / Path(m["modules"][k]["path"]).parent / name).is_file()), None)
        if f is None:
            continue
        w = json.loads(f.read_text())
        rows = []
        for r in w["rows"]:
            row = {"case": r["case"], "dim": r["dim"]}
            for a in s["methods"]:
                if a in r:
                    row[a] = {"seconds": round(r[a]["seconds"], 4)}
                    if "vs_bigstick" in r[a]:
                        row[a]["ref"] = r[a]["vs_bigstick"]
            if "agreement" in r:
                row["agreement"] = r["agreement"]
            rows.append(row)
        s["sweep"] = {"file": str(f.relative_to(root)), "reference": w.get("reference"),
                      "seconds": w.get("seconds"), "rows": rows}


def summary(m):
    lines = 0
    for d in m["defs"]:
        if d["end"] >= d["line"] > 0:
            lines += d["end"] - d["line"] + 1
    mods = sorted({d["module"] for d in m["defs"] if not d.get("main")})
    return {"defs": sum(1 for d in m["defs"] if not d.get("main")), "modules": len(mods), "lines": lines,
            "defsInProgram": m["defsInProgram"]}


_CALL = re.compile(r"\bvv\.([a-z_]+)\.([a-z_][a-z0-9_]*)\s*\(")
_SISSO = re.compile(r"\bsisso\.fit\s*\(")


def gallery_calls(root: Path, quals):
    """gallery slug -> the registered functions its code calls (`vp.<family>.<name>(`)."""
    from . import _gallery
    out = {}
    g = _gallery.gallery_dir()
    if not (g / "index.txt").is_file():
        return out
    for _topic, _section, script in _gallery.read_index(g):
        p = g / script
        text = p.read_text(encoding="utf-8")
        found = []
        for fam, name in _CALL.findall(text):
            q = fam + "." + name
            if q in quals and q not in found:
                found.append(q)
        if _SISSO.search(text) and "sisso.fit" in quals:
            found.append("sisso.fit")
        if found:
            out[p.stem.replace("_", "-")] = found
    return out


def build(root: Path | None = None):
    """{path: text} of every map file and the index."""
    root = root or _spec.repo_root()
    targets = {}
    for qual, reg in _registry.FUNCTIONS.items():
        t = target_of(qual, reg, root)
        if t:
            targets[qual] = t
    for qual, e in EXTRA.items():
        targets[qual] = target_of(qual, e, root)
    maps = {}
    for qual, (file, entry) in targets.items():
        k = key_of(file, entry)
        if k not in maps:
            maps[k] = compile_map(root, file, entry)
    files = {}
    index = {"format": FORMAT, "maps": {}, "functions": {}, "gallery": {}}
    for k, m in sorted(maps.items()):
        text = json.dumps(m, ensure_ascii=False, separators=(",", ":")) + "\n"
        files[map_dir() / (k + ".json")] = text
        s = summary(m)
        s.update({"root": m["root"], "entry": m["entry"], "file": m["modules"][0]["path"],
                  "sha": hashlib.sha256(text.encode()).hexdigest()[:12]})
        index["maps"][k] = s
    for qual, (file, entry) in sorted(targets.items()):
        reg = _registry.FUNCTIONS.get(qual) or EXTRA[qual]
        index["functions"][qual] = {"map": key_of(file, entry), "explore": reg.get("explore"),
                                    "registered": qual in _registry.FUNCTIONS,
                                    "title": reg.get("title"), "about": reg.get("about")}
    index["gallery"] = gallery_calls(root, set(targets))
    files[site_index()] = json.dumps(index, indent=1, ensure_ascii=False) + "\n"
    return files
