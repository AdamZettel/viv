# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The import hook: `import kernels` finds `kernels.viv`.

A finder at the end of `sys.meta_path`, so a .py module or a package of the
same name wins, as an extension module would not. It looks for
`<name>.viv` in each directory of `sys.path` for a top-level import, and in
the package's `__path__` for `from pkg.sub import kernels`.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
from pathlib import Path

from . import _core


class VivLoader(importlib.abc.Loader):
    def __init__(self, path: Path, fast: bool):
        self.path = path
        self.fast = fast

    def create_module(self, spec):
        return None                     # the default module object

    def exec_module(self, module):
        e = _core.build(self.path, fast=self.fast)
        _core.exec_entry(e, module, self.path)

    def get_filename(self, fullname):
        return str(self.path)

    def get_source(self, fullname):
        return self.path.read_text(encoding="utf-8")


class VivFinder(importlib.abc.MetaPathFinder):
    def __init__(self, fast=False):
        self.fast = fast

    def find_spec(self, fullname, path=None, target=None):
        leaf = fullname.rpartition(".")[2]
        dirs = path if path is not None else sys.path
        for d in dirs:
            if not isinstance(d, str):
                continue
            cand = Path(d or os.getcwd()) / (leaf + ".viv")
            if cand.is_file():
                loader = VivLoader(cand.resolve(), self.fast)
                return importlib.machinery.ModuleSpec(fullname, loader, origin=str(cand.resolve()))
        return None


def install(fast=False):
    """Install the finder (once); returns it."""
    for f in sys.meta_path:
        if isinstance(f, VivFinder):
            f.fast = fast
            return f
    f = VivFinder(fast)
    sys.meta_path.append(f)
    return f


def uninstall():
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, VivFinder)]
