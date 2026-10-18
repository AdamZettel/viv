# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The `vipyr` command: the front door.

    vipyr catalog [--topic T] [--json]  the validated engines (vipyr/_agent.py)
    vipyr help [--agent [--brief]]  help; --agent: the guide for an agent
    vipyr run|check|features|explain|map|claims|skill ...
                                    the viv commands, through the bundled compiler
    vipyr stubs kernels.viv [...]   write kernels.pyi beside each source
    vipyr build kernels.viv [--fast] compile into the cache; print the entry
    vipyr compiler                  the viv binary and Viv's library root
    vipyr viv ARGS...               run the bundled viv binary
    vipyr compile model.pkl -o m.c  a fitted scikit-learn pipeline as one C99 file (--data, --features)
"""
from __future__ import annotations

import os
import subprocess
import sys

from . import _agent, _core


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_agent.HELP)
        return 0
    cmd, rest = argv[0], argv[1:]
    try:
        if cmd == "catalog":
            return _agent.catalog(rest)
        if cmd == "help":
            return _agent.help_(rest)
        if cmd in _agent.PASS_THROUGH:
            return _agent.viv([cmd] + rest)
        if cmd == "stubs":
            if not rest:
                print("usage: vipyr stubs FILE.viv ...", file=sys.stderr)
                return 2
            for f in rest:
                print("wrote %s" % _core.write_stub(f))
            return 0
        if cmd == "build":
            fast = "--fast" in rest
            for f in [a for a in rest if a != "--fast"]:
                e = _core.build(f, fast=fast)
                print("%s -> %s" % (f, e.path))
                for n in e.notes:
                    print("note: %s" % n)
            return 0
        if cmd == "compiler":
            print("viv:  %s" % _core.compiler())
            print("root: %s" % _core.viv_root())
            print("id:   %s" % _core.compiler_identity())
            return 0
        if cmd == "compile":
            from .compile import main as compile_main
            return compile_main(rest)
        if cmd == "viv":
            return subprocess.call([_core.compiler()] + rest, env=_core._env())
    except _core.VivCompileError as e:
        print("%s:%s:%s: error: %s" % (e.filename, e.lineno, e.offset, e.msg), file=sys.stderr)
        if e.text:
            print("    " + e.text.rstrip("\n"), file=sys.stderr)
        return 1
    except _core.VivBuildError as e:
        print(str(e), file=sys.stderr)
        return 1
    print("vipyr: unknown command %r (vipyr help lists them)" % cmd, file=sys.stderr)
    return 2


def viv_main(argv=None):
    """The `viv` command a wheel installs: the bundled compiler, run with
    Viv's library root. A `viv` found on PATH could be this script again,
    so a nested call stops instead of recursing."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if os.environ.get("VIPYR_VIV_SHIM"):
        print("vipyr: no `viv` binary: this wheel carries none; set $VIV or install a platform wheel",
              file=sys.stderr)
        return 127
    try:
        viv = _core.compiler()
    except _core.VivBuildError as e:
        print(str(e), file=sys.stderr)
        return 127
    env = dict(_core._env(), VIPYR_VIV_SHIM="1")
    return subprocess.call([viv] + argv, env=env)


if __name__ == "__main__":
    sys.exit(main())
