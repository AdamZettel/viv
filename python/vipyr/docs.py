# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Write vipyr's specs: `python -m vipyr.docs`.

    python -m vipyr.docs            every distribution's spec.json and www/src/data/vipyr.json
    python -m vipyr.docs --check    exit 1 when any file differs from a fresh derivation
    python -m vipyr.docs --gallery-only   the gallery alone (www/src/data/gallery.json)
    python -m vipyr.docs --no-gallery     the specs alone

The code maps (vipyr/_codemap.py) are written with the specs: for each
function, the compiler's map of the Viv program behind it
(`viv graph --map`), in www/public/codemap/ with an index in
www/src/data/codemap.json; /vipyr/map/<function> draws them.

The specs are derived from the manifests and the wrappers `viv export`
writes from them (vipyr/_spec.py). Each distribution ships the part for its
own families (vipyr/spec.json, packages/vipyr-chem/vipyr_chem/spec.json,
...) and reads it for its signatures and docstrings; the site's /vipyr
pages read all of them in one file under www.

The gallery (vipyr/_gallery.py) is built here too: every script listed in
gallery/index.txt is run, and what it printed and drew goes into
www/src/data/gallery.json, so a changed output fails --check like a changed
example does.
"""
import argparse
import os
import sys
from pathlib import Path

import json

from . import _codemap, _gallery, _spec


def site_file() -> Path:
    return _spec.repo_root() / "www" / "src" / "data" / "vipyr.json"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m vipyr.docs")
    ap.add_argument("--check", action="store_true")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--gallery-only", action="store_true")
    g.add_argument("--no-gallery", action="store_true")
    a = ap.parse_args(argv)
    files = {}
    if not a.gallery_only:
        full = _spec.build_all()
        # the examples call the functions as the fresh specs describe them, so
        # a function added since the last run is there to call
        from . import _runtime
        for dist, part in _spec.split(full).items():
            _runtime._specs_cache[str(_spec.spec_file(dist))] = json.loads(_spec.dump(part))
        # every example runs as written; its output is part of the page, so a
        # changed output is a stale file (and fails --check)
        for q, s in full["functions"].items():
            s["examples"] = _spec.run_examples(s) if s["kind"] != "coming" else []
            s.pop("examples_src", None)
        files[site_file()] = _spec.dump(full)
        for dist, part in _spec.split(full).items():
            files[_spec.spec_file(dist)] = _spec.dump(part)
        files.update(_codemap.build())
    if not a.no_gallery:
        files[_gallery.site_file()] = json.dumps(_gallery.build(), indent=1, ensure_ascii=False) + "\n"
    # a map file no function names any more
    left = [] if a.gallery_only or not _codemap.map_dir().is_dir() else \
        [p for p in sorted(_codemap.map_dir().glob("*.json")) if p not in files]
    if a.check:
        stale = [str(t) for t, text in files.items() if not t.is_file() or t.read_text(encoding="utf-8") != text]
        stale += [str(p) + " (no function names it)" for p in left]
        for t in stale:
            print("stale: %s (python -m vipyr.docs rewrites it)" % t, file=sys.stderr)
        return 1 if stale else 0
    for t, text in files.items():
        # an unchanged file is left alone, and a changed one replaced whole:
        # the site's staging (www/scripts/vipyr-build.mjs) runs this while
        # python/tests reads the same files with --check
        if t.is_file() and t.read_text(encoding="utf-8") == text:
            continue
        t.parent.mkdir(parents=True, exist_ok=True)
        tmp = t.with_name(t.name + ".tmp%d" % os.getpid())
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, t)
        print("wrote", t)
    for p in left:
        p.unlink()
        print("removed", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
