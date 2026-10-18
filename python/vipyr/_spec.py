# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Specs: what a vipyr function takes and returns, read from the program.

A spec is a JSON-able dict built from three sources and nothing else:

- the manifest (`<name>.export.toml`): parameter types, ranges, enum labels,
  shapes, groups, results, decks, the stated reference and tolerance;
- the wrapper `viv export` writes from it: its docstring carries the
  program's summary (the opening paragraph of its intent lens or `#|`
  block), each parameter's `#:` help and default (per deck), and each
  result's meaning;
- vipyr/_registry.py: the Python name, the field, renamed keywords, the
  example.

`build_all()` derives every spec; `python -m vipyr.docs` writes them to
vipyr/spec.json (what import-time signatures and docstrings use) and to the
site's www/src/data/vipyr.json. tests/test_vipyr_programs.py checks the stored file
against a fresh derivation.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

try:
    import tomllib as _toml
except ImportError:                      # Python 3.9, 3.10
    try:
        import tomli as _toml
    except ImportError:                  # pragma: no cover
        _toml = None

from vipyr import _core

from . import _registry

SPEC_FORMAT = 1
HERE = Path(__file__).resolve().parent
SPEC_FILE = HERE / "spec.json"

# fields every result carries; a program result of the same name is listed as program_<name>
RESERVED_RESULTS = ("params", "tests", "log", "run_id", "record")

_PY_TYPE = {"Num": "float", "Int": "int", "Bool": "bool", "Complex": "complex", "Cplx": "complex",
            "Arr": "ndarray[float64]", "[Int]": "ndarray[int64]", "IntArr": "ndarray[int64]", "Str": "str"}

# Units as the programs' `#:` lines spell them, longest first; a help text
# names its unit after its last comma ("plane-wave cutoff, hartree") or in
# parentheses at its end ("(hartree)").
_UNITS = ["MeV fm^(3 + 3 ALPHA)", "MeV fm^3", "MeV fm^5", "fm^-3", "fm^-4", "fm^-5", "hartree", "MeV", "eV",
          "fm", "bohr", "angstrom", "units of e", "units of the plasma frequency",
          "units of the inverse plasma frequency"]


def repo_root() -> Path:
    r = _core._repo_root() or (Path(_core.viv_root()) if _core.viv_root() else None)
    if r is None:
        raise RuntimeError("vipyr: the programs' sources are not found: run from a checkout of the Viv "
                           "repository or set $VIV_ROOT to one")
    return Path(r)


def source(rel) -> Path:
    """A program's file by its repository-relative path: in the checkout
    when there is one, else in the `_root` of an installed distribution
    (a platform wheel ships its programs' sources there; vipyr/_prebuilt.py).
    The checkout's path is returned when the file is found nowhere, so the
    error names it."""
    from . import _prebuilt
    r = _core._repo_root()
    if r is not None and (r / rel).exists():
        return r / rel
    for root in _prebuilt.roots():
        if (root / rel).exists():
            return root / rel
    return repo_root() / rel


def read_toml(path: Path) -> dict:
    if _toml is None:
        raise RuntimeError("vipyr: reading manifests needs Python 3.11 or the tomli package")
    with open(path, "rb") as f:
        return _toml.load(f)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def units_of(text: str):
    if not text:
        return None
    t = text.strip().rstrip(".")
    m = re.search(r"\(([^()]*)\)$", t)
    for u in _UNITS:
        if m and m.group(1) == u:
            return u
        if t.endswith(", " + u) or re.search(r", %s(?:[:;,]| \()" % re.escape(u), t) or t.endswith("in " + u):
            return u
    return None


# ------------------------------------------------------------- docstrings

def _whole_sentences(text: str) -> str:
    """A summary cut at a paragraph break can end mid-sentence (a sentence
    that runs on into a displayed formula: "By the Green-Kubo relations,").
    Such a tail is dropped, back to the last full stop."""
    t = text.rstrip()
    if not t or t[-1] not in ",:;(":
        return t
    k = max(t.rfind(". "), t.rfind("? "), t.rfind("! "))
    return t[:k + 1] if k > 0 else t


def _split_params_block(doc: str):
    """(summary, parameter lines, notes, result lines) of a generated
    program docstring."""
    lines = doc.splitlines()
    i = next((k for k, l in enumerate(lines) if l.startswith("Parameters (")), len(lines))
    summary = _whole_sentences(" ".join(l.strip() for l in lines[:i] if l.strip()))
    j = next((k for k, l in enumerate(lines) if l.startswith("log: a list")), len(lines))
    params = lines[i + 1:j]
    # results: after the "Returns a dict" paragraph, up to the version line
    k = j
    while k < len(lines) and lines[k].strip():
        k += 1
    res = [l for l in lines[k + 1:] if l.strip() and not l.startswith("viv ")]
    return summary, params, res


def _deck_values(text: str, decks):
    if not decks:
        return text
    pat = r"(?:^|, )(%s) " % "|".join(re.escape(d) for d in decks)
    parts = re.split(pat, text)
    out = {}
    for n in range(1, len(parts) - 1, 2):
        out[parts[n]] = parts[n + 1]
    return out if out else text


def parse_program_doc(doc: str, decks=()):
    summary, plines, rlines = _split_params_block(doc)
    params, notes, cur = {}, [], None
    for l in plines:
        if not l.strip():
            continue
        if l.startswith("    "):
            s = l.strip()
            if cur is not None and s.startswith("default: "):
                cur["default"] = _deck_values(s[len("default: "):], decks)
            elif cur is not None:
                cur["note"] = (cur.get("note", "") + " " + s).strip()
            continue
        m = re.match(r"^(\w+) \((.*?)\): (.*)$", l)
        if m:
            cur = params[m.group(1)] = {"shown_type": m.group(2), "help": m.group(3)}
        elif re.match(r"^(\w+): one of ", l):
            cur = None
        else:
            notes.append(l.strip())
            cur = None
    results = {}
    for l in rlines:
        m = re.match(r"^(\w+): (.*)$", l)
        if m:
            results[m.group(1)] = m.group(2)
    return summary, params, notes, results


def parse_kernel_doc(doc: str):
    """The summary paragraph of a kernel's docstring (after its signature
    line), with the reference and tolerance paragraphs left to the manifest."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", doc.strip())]
    body = [p for p in paras[1:] if not re.match(r"^(\w+ \(|Returns|reference:|tolerance:)", p)]
    return _whole_sentences(" ".join(" ".join(p.split()) for p in body[:1]))


def docstrings_of(py_file: Path) -> dict:
    """{function: docstring} of a generated wrapper, read without importing
    it (no library is built)."""
    tree = ast.parse(Path(py_file).read_text(encoding="utf-8"))
    return {n.name: ast.get_docstring(n) or "" for n in tree.body if isinstance(n, ast.FunctionDef)}


def _kernels_table(py_file: Path) -> dict:
    tree = ast.parse(Path(py_file).read_text(encoding="utf-8"))
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "KERNELS" for t in n.targets):
            return ast.literal_eval(n.value)
    return {}


def export_to(src: Path, out: Path):
    """`viv export` of a manifest (or `--infer` of a .viv file) into `out`:
    the wrapper only, no C compiler involved."""
    src = Path(src)
    m = _core.manifest_for(src)
    cmd = [_core.compiler(), "export"] + ([str(m)] if m else ["--infer", str(src)]) + ["-o", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=_core._env())
    if r.returncode != 0:
        raise _core.compile_error(r.stderr, src)
    py = [p for p in Path(out).glob("*.py") if not p.name.startswith("test_")]
    return py[0]


# ---------------------------------------------------------------- params

def _param_entry(key, p, info, decks):
    t = p.get("type", "Num")
    labels = p.get("labels") or (p.get("values") if t == "Enum" else None)
    e = {"name": key.lower(), "manifest_name": key, "type": t,
         "python_type": "str" if labels else _PY_TYPE.get(t, t)}
    for k in ("min", "max", "min_exclusive", "max_exclusive", "rank", "shape", "square", "nonfinite", "normalize"):
        if k in p:
            e[k] = p[k]
    if labels:
        e["choices"] = list(labels)
    if p.get("contract"):
        e["contract"] = p["contract"]
    help_ = p.get("doc") or info.get("help", "")
    e["help"] = help_
    if info.get("note"):
        e["note"] = "case does not matter" if info["note"] == "in any case" else info["note"]
    e["units"] = units_of(help_)
    e["default"] = info.get("default")
    if isinstance(e["default"], str) and len(e["default"]) > 24 and array_shape(e["default"]):
        e["default_shape"] = list(array_shape(e["default"]))
    e["range"] = range_text(e)
    return e


def range_text(e) -> str:
    if e.get("choices"):
        return "one of " + ", ".join(e["choices"])
    lo = e.get("min", e.get("min_exclusive"))
    hi = e.get("max", e.get("max_exclusive"))
    lo_s = None if lo is None else ("> %s" if "min_exclusive" in e else ">= %s") % _num(lo)
    hi_s = None if hi is None else ("< %s" if "max_exclusive" in e else "<= %s") % _num(hi)
    parts = [s for s in (lo_s, hi_s) if s]
    out = ", ".join(parts)
    if e.get("type") in ("Arr", "[Int]", "IntArr") and parts:
        out = "each element " + out
    if e.get("shape"):
        out = (out + "; " if out else "") + "shape (%s)" % ", ".join(
            "len(%s)" % s.lower() if isinstance(s, str) else str(s) for s in e["shape"])
    elif e.get("rank"):
        out = (out + "; " if out else "") + "rank %s" % e["rank"] + (", square" if e.get("square") else "")
    return out


def _num(v):
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


# ----------------------------------------------------------------- specs

def source_help(viv: Path) -> dict:
    """{name: help} from the `#: name  text` blocks of a .viv file, with
    their `#:     ...` continuation lines (the wrapper's docstring keeps the
    first line only)."""
    out, cur = {}, None
    for line in viv.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^#: (\w+)\s{2,}(.*)$", line)
        if m:
            cur = m.group(1).lower()
            out[cur] = m.group(2).strip()
        elif cur and re.match(r"^#:\s{3,}\S", line):
            out[cur] += " " + line[2:].strip()
        else:
            cur = None
    return out


def _program(qual, reg, root, tmp):
    mpath = root / reg["manifest"]
    man = read_toml(mpath)
    py = export_to(mpath, tmp / qual)
    decks = list(man.get("decks", {}))
    summary, pinfo, notes, rinfo = parse_program_doc(docstrings_of(py)["run"], decks)
    aliases = reg.get("aliases", {})
    inv = {v: k for k, v in aliases.items()}
    params = []
    deck_param = man.get("deck_param", "material")
    if decks:
        params.append({"name": deck_param, "manifest_name": deck_param, "type": "deck", "python_type": "str",
                       "choices": decks, "default": decks[0], "help": "the deck: a named set of every "
                       "parameter's value, with the checks that hold at it", "units": None,
                       "range": "one of " + ", ".join(decks)})
    full = {}
    for f in man.get("files", []):
        if (root / f).is_file():
            full.update(source_help(root / f))
    for key, p in man.get("params", {}).items():
        info = dict(pinfo.get(key.lower(), {}))
        longer = full.get(key.lower())
        if longer and info.get("help") and longer.startswith(info["help"].rstrip(";")) and len(longer) > len(info["help"]):
            info["help"] = longer
        e = _param_entry(key, p, info, decks)
        if e["name"] in inv:
            e["alias_of"] = e["name"]
            e["name"] = inv[e["name"]]
        params.append(e)
    results = []
    for key, r in man.get("results", {}).items():
        r = {"type": r} if isinstance(r, str) else r
        doc = r.get("doc") or rinfo.get(key.lower(), "")
        name = key.lower()
        if name in RESERVED_RESULTS:
            name = "program_" + name
        e = {"name": name, "key": key.lower(), "type": r.get("type", "Num"), "python_type": _PY_TYPE.get(r.get("type"), "object"),
             "doc": doc, "units": units_of(doc.split(" (rank")[0].split(":")[0] if doc else doc) or units_of(doc)}
        if isinstance(r.get("rank"), dict):
            e["rank_by"] = r["rank"]
        results.append(e)
    hashes = mpath.with_name(mpath.name.replace(".export.toml", ".exact_hashes.json"))
    if not hashes.is_file() and man.get("name"):
        hashes = mpath.with_name(man["name"] + ".exact_hashes.json")
    ref = man.get("reference")
    return {
        "summary": summary, "params": params, "results": results, "notes": notes,
        "decks": decks, "deck_param": deck_param if decks else None,
        "groups": man.get("groups", {}),
        "reference": ref,
        "exact_hashes": json.loads(hashes.read_text()) if hashes.is_file() else {},
        "hashes_file": str(hashes.relative_to(root)) if hashes.is_file() else None,
        "sources": man.get("files", []), "about": man.get("about"),
        "cases_file": man.get("cases"),
        "manifest_sha256": sha(mpath),
    }


def _kernel(qual, reg, root, tmp, cache):
    mpath = root / reg["manifest"]
    if mpath not in cache:
        cache[mpath] = (read_toml(mpath), export_to(mpath, tmp / mpath.name))
    man, py = cache[mpath]
    k = man["kernels"][reg["entry"]]
    doc = docstrings_of(py)[reg["entry"]]
    params = []
    for a in k["args"]:
        e = _param_entry(a["name"], a, {}, ())
        e["name"] = a["name"]
        e["default"] = None
        e["required"] = True
        params.append(e)
    results = [{"name": r["name"], "type": r["type"], "python_type": _PY_TYPE.get(r["type"], "object"),
                "doc": ("shape " + r["shape"]) if r.get("shape") else "", "units": None} for r in k["results"]]
    hashes = mpath.with_name(mpath.name.replace(".export.toml", ".exact_hashes.json"))
    hv = json.loads(hashes.read_text()).get(reg["entry"]) if hashes.is_file() else None
    src_file = k["viv"].split(":")[0]
    return {
        "summary": parse_kernel_doc(doc), "params": params, "results": results, "notes": [],
        "reference": {"program": k.get("reference"), "compare": k.get("compare"), "domain": k.get("domain"),
                      "tolerance": k.get("tolerance"), "why": k.get("why")},
        "exact_hashes": {"cases": hv} if hv else {},
        "hashes_file": str(hashes.relative_to(root)) if hashes.is_file() else None,
        "cases_file": man.get("cases"),
        "sources": [src_file] + [s.strip().split(":")[0] for s in k.get("source", "").split(",")[:1]],
        "manifest_sha256": sha(mpath),
    }


def _inferred(qual, reg, root, tmp, cache):
    src = root / reg["source"]
    if src not in cache:
        cache[src] = export_to(src, tmp / src.stem)
    py = cache[src]
    args = _kernels_table(py)[reg["entry"]][0]
    doc = docstrings_of(py)[reg["entry"]]
    params = []
    for a in args:
        t = {"Cplx": "Complex", "IntArr": "[Int]"}.get(a[1], a[1])
        e = {"name": a[0], "manifest_name": a[0], "type": t, "python_type": _PY_TYPE.get(t, t), "help": "",
             "units": None, "default": None, "required": True}
        e["range"] = range_text(e)
        params.append(e)
    results = [{"name": n, "type": t, "python_type": _PY_TYPE.get(t, t), "doc": d, "units": None}
               for n, t, d in reg["results"]]
    return {"summary": parse_kernel_doc(doc), "params": params, "results": results, "notes": [],
            "reference": {"program": reg["checked"]}, "exact_hashes": {}, "hashes_file": None,
            "sources": [reg["source"]], "manifest_sha256": sha(src)}


def _python(qual, reg, root):
    """A Python function of vipyr (kind "python"): its parameters and results
    are listed in the registry; the summary is the docs file's."""
    params = []
    for name, ptype, default, help_ in reg["params"]:
        e = {"name": name, "manifest_name": name, "type": ptype, "python_type": ptype, "help": help_,
             "units": None, "default": default, "required": default is None}
        if name in reg.get("positional", ()):
            e["positional"] = True
        e["range"] = range_text(e)
        params.append(e)
    results = [{"name": n, "type": t, "python_type": t, "doc": d, "units": None} for n, t, d in reg["results"]]
    h = hashlib.sha256()
    for src in reg["sources"]:
        h.update((root / src).read_bytes())
    out = {"summary": reg.get("summary", ""), "params": params, "results": results, "notes": [],
           "reference": {"program": reg["checked"]}, "exact_hashes": {}, "hashes_file": None,
           "sources": list(reg["sources"]), "manifest_sha256": h.hexdigest()}
    if reg.get("raises"):
        out["raises_python"] = [list(r) for r in reg["raises"]]
    if reg.get("unpack"):
        out["unpack"] = reg["unpack"]
    return out


def _feature(qual, reg, root):
    import sys
    sys.path.insert(0, str(root / "catalog"))
    import catalog
    e = catalog.ENTRIES[reg["entry"]]
    import inspect
    sig = inspect.signature(e.fn)
    params = [{"name": "df", "manifest_name": "df", "type": "DataFrame", "python_type": "pandas.DataFrame",
               "help": "the event table, rows in any order; None makes a synthetic one of n rows",
               "units": None, "default": "None", "range": ""},
              {"name": "n", "manifest_name": "n", "type": "Int", "python_type": "int",
               "help": "rows of the synthetic table when df is None", "units": None, "default": "500",
               "range": ">= 1"}]
    for name, p in list(sig.parameters.items())[1:]:
        params.append({"name": name, "manifest_name": name, "type": "Str", "python_type": "str", "help": "",
                       "units": None, "default": repr(p.default), "range": ""})
    readme = (root / "catalog" / "README.md").read_text()
    m = re.search(r"### \d+\. %s \(`[^`]+`\)\n(.*?)(?=\n### |\Z)" % re.escape(reg["entry"]), readme, re.S)
    section = m.group(1) if m else ""
    summary = " ".join((getattr(catalog, reg["entry"]).__doc__ or "").split())
    outs = re.findall(r"^\| `(\w+)` \| (.*?) \|$", section, re.M)
    results = [{"name": "frame", "type": "DataFrame", "python_type": "pandas.DataFrame",
                "doc": "the features, one row per input row, in the input's order and index: "
                       + ", ".join(o[0] for o in outs), "units": None}]
    return {"summary": summary, "params": params, "results": results, "notes": [],
            "outputs": [{"name": o[0], "doc": o[1]} for o in outs],
            "reference": {"program": "catalog.%s_reference: a pandas groupby().apply() with a Python loop per "
                          "row, equal %s; vipyr.check_point_in_time: no feature reads a row after its own "
                          "time" % (reg["entry"], "bit for bit" if e.exact else "to 1e-13 relative")},
            "exact_hashes": {}, "hashes_file": None, "sources": [reg["source"], "catalog/catalog.py"],
            "manifest_sha256": sha(root / reg["source"])}


REF_ORDER = ["program", "cases", "compare", "domain"]


def ref_items(spec):
    """The reference table as (label, text) pairs, in reading order."""
    ref = spec.get("reference") or {}
    keys = [k for k in REF_ORDER if ref.get(k)] + sorted(
        k for k in ref if k not in REF_ORDER + ["tolerance", "why"] and ref[k]) + [
        k for k in ("tolerance", "why") if ref.get(k)]
    out = []
    for k in keys:
        v = ref[k]
        if isinstance(v, dict):
            v = ", ".join("%s %s" % (a, b) for a, b in v.items())
        out.append((k, str(v)))
    return out


def signature_text(qual, spec) -> str:
    parts, star = [], False
    for p in spec["params"]:
        if p.get("required"):
            parts.append(p["name"])
        else:
            if not star and spec.get("kind") in ("program", "python") and p["type"] != "deck" \
                    and not p.get("positional"):
                parts.append("*")
                star = True
            d = p.get("default")
            if p["type"] == "deck":
                parts.append('%s="%s"' % (p["name"], d))
            else:
                parts.append("%s=%s" % (p["name"], short_default(d)))
    return "vp.%s(%s)" % (qual, ", ".join(parts))


def short_default(d) -> str:
    """A default as a signature shows it: the program's value when it is
    short, `...` for a long array (the parameter's entry gives its shape),
    None for a per-deck value."""
    if d is None or isinstance(d, dict):
        return "None"
    d = str(d)
    return d if len(d) <= 24 else "..."


def array_shape(d):
    """(rows, cols) of a default written as a nested list, else None."""
    try:
        v = json.loads(d) if isinstance(d, str) else None
    except ValueError:
        return None
    if isinstance(v, list) and v and all(isinstance(r, list) for r in v):
        return (len(v), len(v[0]))
    return None


# ------------------------------------------------------------ authored docs

DOC_KEYS = ("summary", "description", "notes", "references", "see_also", "examples")


def docs_file(reg, root: Path):
    """The authored reference text of a function: `reg["docs"]` if the
    registry names one, else `<name>.docs.toml` beside a program's manifest.
    (`viv export` reads the manifest strictly, so the prose lives beside it.)"""
    if reg.get("docs"):
        return root / reg["docs"]
    if reg.get("kind") == "program" and reg.get("manifest"):
        p = root / reg["manifest"].replace(".export.toml", ".docs.toml")
        return p if p.is_file() else None
    return None


def authored(qual, reg, root: Path) -> dict:
    """The docs file's entries for one function: top-level keys for a
    program, the `[<entry>]` table for a kernel or a def of a shared file.

        summary      one line, in place of the program's own
        description  paragraphs (markdown)
        notes        the method (markdown with $...$ and $$...$$ math)
        references   [{text, doi | url}], upstream code first for a port
        see_also     ["family.name", ...]
        examples     [{code, caption}], run by `python -m vipyr.docs`
        [params.<name>]  doc, units, range (what happens at the edges)
    """
    p = docs_file(reg, root)
    if p is None:
        return {}
    d = read_toml(p)
    if reg.get("kind") != "program":
        d = d.get(reg.get("entry") or qual.split(".")[1], {})
    unknown = set(d) - set(DOC_KEYS) - {"params"}
    if unknown:
        raise ValueError("%s: unknown keys %s (known: %s, params)" % (p, sorted(unknown), ", ".join(DOC_KEYS)))
    d["_file"] = str(p.relative_to(root))
    return d


def raises(spec) -> list:
    """What a call raises, from the kind of function and its parameters."""
    if spec["kind"] == "coming":
        return [("NotImplementedError", "always: the function is coming soon, listed but not built")]
    out = []
    if any(p.get("choices") or any(k in p for k in ("min", "max", "min_exclusive", "max_exclusive", "rank", "shape"))
           for p in spec["params"]) or spec.get("groups"):
        out.append(("vipyr.ParameterError", "a ValueError: an argument outside its range, choices, rank or shape, "
                    "or one of a group given without the rest; its .param names the argument"))
    if spec["kind"] == "python" and spec.get("raises_python"):
        out.extend(tuple(r) for r in spec["raises_python"])
    elif spec["kind"] == "python":
        out.append(("ValueError", "a method, interval, tolerance or array shape the solver cannot take; "
                    "vipyr._ode.OdeProblemError (a ValueError) when the rhs, events or jac cannot be made "
                    "into a Viv problem, or py2viv refuses a Python rhs"))
        out.append(("vipyr.VivCompileError", "a SyntaxError: the rhs's .viv file does not compile"))
    out.append(("TypeError", "an argument of the wrong type, or an unknown keyword"))
    out.append(("vipyr.VivBuildError", "an ImportError: no C compiler, or the program's C does not build"))
    if spec["kind"] == "program":
        out.append(("RuntimeError", "the program stops with its own message (a combination it rejects, "
                    "or no convergence where the program treats that as an error)"))
    return out


def run_examples(spec) -> list:
    """Each example run as written, its printed output kept beside it."""
    import contextlib
    import io
    out = []
    for ex in spec.get("examples_src", []):
        buf = io.StringIO()
        name = "<example of vp.%s>" % spec["qualname"]
        # registered as a source, so inspect.getsource works in it (vp.translate reads its function's source)
        import linecache
        linecache.cache[name] = (len(ex["code"]), None, ex["code"].splitlines(True), name)
        with contextlib.redirect_stdout(buf):
            exec(compile(ex["code"], name, "exec"), {"__name__": "__example__"})
        out.append({"code": ex["code"], "caption": ex.get("caption"), "output": buf.getvalue().rstrip("\n")})
    return out


def build_all(only=None) -> dict:
    root = repo_root()
    out = {"format": SPEC_FORMAT,
           "fields": [{"id": f, "title": t, "about": a, "overview": _registry.OVERVIEWS.get(f),
                       "dist": _registry.FAMILY_DIST[f],
                       "extra": _registry.DISTS[_registry.FAMILY_DIST[f]]["extra"]} for f, t, a in _registry.FIELDS],
           "dists": [dict(name=n, **d) for n, d in _registry.DISTS.items()],
           "functions": {}}
    cache = {}
    with tempfile.TemporaryDirectory(prefix="vipyr-spec-") as tmp:
        tmp = Path(tmp)
        for qual, reg in _registry.FUNCTIONS.items():
            if only and qual not in only:
                continue
            kind = reg["kind"]
            if kind == "program":
                s = _program(qual, reg, root, tmp)
            elif kind == "kernel":
                s = _kernel(qual, reg, root, tmp, cache)
            elif kind == "inferred":
                s = _inferred(qual, reg, root, tmp, cache)
            elif kind == "feature":
                s = _feature(qual, reg, root)
            elif kind == "python":
                s = _python(qual, reg, root)
            else:
                s = {"summary": reg["summary"], "params": [], "results": [], "notes": [], "where": reg["where"]}
            family, name = qual.split(".")
            s.update({"qualname": qual, "family": family, "name": name, "kind": kind,
                      "result": reg.get("result") or _class_name(name),
                      "aliases": reg.get("aliases", {}), "example": reg.get("example"),
                      "explore": reg.get("explore"), "manifest": reg.get("manifest"),
                      "dist": _registry.FAMILY_DIST[family],
                      "extra": _registry.DISTS[_registry.FAMILY_DIST[family]]["extra"]})
            doc = authored(qual, reg, root)
            if doc.get("summary"):
                s["summary"] = doc["summary"]
            s["description"] = doc.get("description")
            s["notes_md"] = doc.get("notes")          # "notes" is the program's own list of lines
            s["references"] = doc.get("references", [])
            s["see_also"] = doc.get("see_also", [])
            s["docs_file"] = doc.get("_file")
            for prm in s["params"]:
                extra = doc.get("params", {}).get(prm["name"], {})
                if extra.get("doc"):
                    prm["help"] = extra["doc"]
                if extra.get("units"):
                    prm["units"] = extra["units"]
                if extra.get("range"):
                    prm["range_doc"] = extra["range"]
            s["examples_src"] = ([{"code": reg["example"], "caption": None}] if reg.get("example") else []) + \
                list(doc.get("examples", []))
            s["raises"] = [{"name": n, "text": t} for n, t in raises(dict(s, kind=kind))]
            s["signature"] = signature_text(qual, s) if kind != "coming" else "vp.%s(...)" % qual
            s["checked"] = [{"label": k, "text": v} for k, v in ref_items(s)]
            out["functions"][qual] = s
    return out


def _class_name(name):
    return "".join(w.capitalize() for w in name.split("_")) + "Result"


def spec_file(dist: str) -> Path:
    """Where a distribution keeps its spec.json: vipyr/spec.json for the core,
    <path>/<module>/spec.json under the repository for the others."""
    if dist == "vipyr":
        return SPEC_FILE
    d = _registry.DISTS[dist]
    return repo_root() / d["path"] / d["module"] / "spec.json"


def split(specs) -> dict:
    """{dist: the specs it ships}: each holds its own families' fields and functions."""
    out = {}
    for dist in _registry.DISTS:
        fams = {f["id"] for f in specs["fields"] if f["dist"] == dist}
        if not fams:
            continue
        out[dist] = {"format": specs["format"], "dist": dist,
                     "fields": [f for f in specs["fields"] if f["id"] in fams],
                     "functions": {q: v for q, v in specs["functions"].items() if v["family"] in fams}}
    return out


def load_specs() -> dict:
    """Every distribution's stored specs in one (the checkout's view):
    the site's copy, www/src/data/vipyr.json, holds the same."""
    with open(SPEC_FILE, encoding="utf-8") as f:
        out = json.load(f)
    out["functions"] = dict(out["functions"])
    for dist in _registry.DISTS:
        if dist == "vipyr":
            continue
        p = spec_file(dist)
        if p.is_file():
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            out["fields"] = out["fields"] + d["fields"]
            out["functions"].update(d["functions"])
    return out


def dump(specs) -> str:
    return json.dumps(specs, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
