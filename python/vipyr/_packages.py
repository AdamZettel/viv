# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Which families beyond the core are available, and where they come from.

vipyr is the core distribution: the tools and the broadly useful numerics
(special, linalg, finance, features, statmech, randla). The domain families
ship as their own distributions (vipyr-chem for scf and dft, vipyr-nuclear,
vipyr-plasma, vipyr-xc, ...), each registering its families in the
entry-point group `vipyr.packages`:

    [project.entry-points."vipyr.packages"]
    scf = "vipyr_chem.scf"

`vp.scf` then imports `vipyr_chem.scf` on first use. A family whose
distribution is not installed raises an ImportError naming it:
"vp.scf needs vipyr-chem: pip install vipyr[chem]".

In a checkout of the Viv repository nothing has to be installed: the
directories `packages/*/` and `ports/*/` (not `_template`) are read the same
way, from their pyproject.toml, and put on sys.path. `VIPYR_DEV=0` turns
that off, so the checkout behaves like a core-only install.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from . import _registry

GROUP = "vipyr.packages"
_HERE = Path(__file__).resolve().parent
_found: dict = {}


def checkout_root():
    """The repository root when this vipyr is python/vipyr of a checkout."""
    root = _HERE.parents[1]
    if _HERE.parent.name == "python" and (root / "packages").is_dir():
        return root
    return None


def checkout_dirs():
    """The distribution directories of the checkout: packages/* and ports/*."""
    root = checkout_root()
    if root is None:
        return []
    out = []
    for top in ("packages", "ports"):
        d = root / top
        if d.is_dir():
            out += [p for p in sorted(d.iterdir())
                    if p.is_dir() and not p.name.startswith(("_", ".")) and (p / "pyproject.toml").is_file()]
    return out


def _checkout_entry_points():
    if os.environ.get("VIPYR_DEV") == "0":
        return {}
    from .build import read_pyproject
    eps = {}
    for d in checkout_dirs():
        data = read_pyproject(d / "pyproject.toml")
        group = data.get("project", {}).get("entry-points", {}).get(GROUP, {})
        if group and str(d) not in sys.path:
            sys.path.append(str(d))
        for name, target in group.items():
            eps.setdefault(name, target)
    return eps


def _installed_entry_points():
    try:
        from importlib.metadata import entry_points
    except ImportError:                                 # Python < 3.8
        return {}
    try:
        eps = entry_points(group=GROUP)                 # Python 3.10+
    except TypeError:
        eps = entry_points().get(GROUP, [])             # Python 3.9
    return {e.name: e.value for e in eps}


def registered() -> dict:
    """{family: module} of every family beyond the core: the checkout's
    first (a checkout tests its own sources), then installed distributions."""
    if "eps" not in _found:
        eps = _checkout_entry_points()
        for k, v in _installed_entry_points().items():
            eps.setdefault(k, v)
        _found["eps"] = eps
    return _found["eps"]


def missing_message(family: str):
    """The ImportError text for a known family whose distribution is absent,
    or None when the name is no family vipyr knows."""
    dist = _registry.FAMILY_DIST.get(family)
    if dist is None or dist == "vipyr":
        return None
    extra = _registry.DISTS[dist].get("extra")
    how = "pip install vipyr[%s]" % extra if extra else "pip install %s" % dist
    return "vp.%s needs %s: %s" % (family, dist, how)
