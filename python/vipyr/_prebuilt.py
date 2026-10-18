# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Precompiled libraries shipped inside platform wheels.

A platform wheel of vipyr (and of each distribution under packages/ and
ports/) is built with its programs already compiled: `vipyr.build` runs the
same export and C build as `vipyr.load` for every program the distribution
carries (`[tool.vipyr] prebuild`), and stores each cache entry under
`<package>/_prebuilt/<dir>/`, with the program's sources under
`<package>/_root/` laid out as in the repository.

`<package>/_prebuilt/index.json` keys each entry by content, not by path:

    key       SHA-256 of the source's bytes, its manifest's bytes and the build
              kind (exact or fast)
    compiler  the identity of the `viv` binary that exported it
    deps      every module the export read, relative to the source's directory
              (or `lib:` + a path under Viv's library root), with its SHA-256
    absent    modules the export looked for and did not find, the same way

`vipyr.load` (through `_core.build`) takes a precompiled entry when the key
matches, the compiler is the one running now, every dep still hashes the
same and nothing absent has appeared; otherwise it compiles as before (a
checkout with edited sources, a user's own .viv, another compiler). So a
precompiled library is used exactly when compiling now would produce it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

FORMAT = "1"
_HERE = Path(__file__).resolve().parent
_memo: dict = {}
stats = {"prebuilt_hits": 0}


def content_key(src: Path, fast: bool) -> str:
    """The lookup key: the source, its manifest and the build kind."""
    from ._core import manifest_for
    h = hashlib.sha256(("vipyr prebuilt %s %s\n" % (FORMAT, "fast" if fast else "exact")).encode())
    h.update(src.read_bytes())
    m = manifest_for(src)
    if m is not None and m != src:
        h.update(b"\0manifest\0" + m.read_bytes())
    return h.hexdigest()


def _package_dirs():
    """vipyr itself, then the top-level package of every distribution the
    entry points name (its `_root` and `_prebuilt` sit beside its modules)."""
    out = [_HERE]
    try:
        from . import _packages
        for target in _packages.registered().values():
            top = target.split(".")[0]
            import importlib.util
            spec = importlib.util.find_spec(top)
            if spec is not None and spec.submodule_search_locations:
                for loc in spec.submodule_search_locations:
                    p = Path(loc).resolve()
                    if p not in out:
                        out.append(p)
    except Exception:          # discovery must never break a load
        pass
    return out


def roots():
    """Every `_root` directory of the installed distributions: where a
    wheel keeps its programs' sources, laid out as in the repository."""
    return [d / "_root" for d in _package_dirs() if (d / "_root").is_dir()]


def stores():
    """{content key: (entry dir, entry record)} over every installed
    `_prebuilt/index.json`."""
    if "stores" not in _memo:
        idx = {}
        for d in _package_dirs():
            f = d / "_prebuilt" / "index.json"
            if not f.is_file():
                continue
            try:
                data = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if data.get("format") != FORMAT:
                continue
            for e in data.get("entries", []):
                idx.setdefault(e["key"], (d / "_prebuilt" / e["dir"], e))
        _memo["stores"] = idx
    return _memo["stores"]


def _resolve(rel: str, src: Path):
    if rel.startswith("lib:"):
        # Viv's library root first (the compiler reads it there), then the
        # `_root/lib` of a distribution whose program lives under lib/
        from ._core import viv_root
        cands = ([Path(viv_root())] if viv_root() else []) + roots()
        for r in cands:
            if (r / "lib" / rel[4:]).exists():
                return r / "lib" / rel[4:]
        return Path(cands[0]) / "lib" / rel[4:] if cands else None
    return (src.parent / rel)


def _sha(p: Path) -> str:
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def lookup(src: Path, fast: bool):
    """The precompiled cache entry for `src`, or None when there is none or
    compiling now could give something else."""
    idx = stores()
    if not idx:
        return None
    hit = idx.get(content_key(src, fast))
    if hit is None:
        return None
    d, e = hit
    from . import _core
    try:
        if e.get("compiler") != _core.compiler_identity():
            return None
    except _core.VivBuildError:
        return None
    for rel, sha in e.get("deps", {}).items():
        p = _resolve(rel, src)
        try:
            if p is None or _sha(p) != sha:
                return None
        except OSError:
            return None
    for rel in e.get("absent", []):
        p = _resolve(rel, src)
        if p is not None and p.exists():
            return None
    info = dict(e["info"], source=str(src), key=e["key"][:20], deps={}, absent=[], prebuilt=True)
    ent = _core.Entry(d, info)
    if not (ent.py.is_file() and ent.lib.is_file()):
        return None
    stats["prebuilt_hits"] += 1
    return ent


def relative_dep(p: str, src: Path, lib_root):
    """How an index records a dep: `lib:<path>` under Viv's library root,
    else relative to the source's directory."""
    p = os.path.realpath(p)
    if lib_root:
        lib = os.path.realpath(os.path.join(str(lib_root), "lib"))
        if p.startswith(lib + os.sep):
            return "lib:" + Path(os.path.relpath(p, lib)).as_posix()
    return Path(os.path.relpath(p, os.path.realpath(str(src.parent)))).as_posix()
