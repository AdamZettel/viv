# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The digits lens from Python: which digits of a program's printed numbers
are real.

    report = vp.digits("examples/digits/hilbert.viv", sets={"n": 8})
    report.stable          # an array: the stable digits of each printed number
    print(report)          # the table `viv digits` prints

`vp.digits` runs `viv digits --json` (lang/viv/src/digits.rs): the program
in the interpreter, once as written, then with its floating-point evaluation
perturbed -- reduction order, FMA contraction, Monte Carlo arithmetic
(Verificarlo's method) -- and per printed number the leading significant
digits that stay the same, with the source lines that move the rest. A
program result's `.digits()` does the same for the program behind it, with
the parameters of that run.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import tempfile

import numpy as np

from . import _core, _spec

__all__ = ["digits", "DigitsReport", "Output"]


@dataclasses.dataclass(frozen=True)
class Output:
    """One printed number: its value, the line it was printed on, and its
    stable digits under each perturbation (None when that one was off)."""
    value: float
    text: str
    order: float | None
    fma: float | None
    mca: float | None
    stable: float


@dataclasses.dataclass(frozen=True)
class DigitsReport:
    """What `viv digits` found. `outputs` in print order; `lines` every
    source line that ran perturbable operations (`at`, `code`, counts, and
    `effects` for the lines perturbed alone); `failed` the perturbed runs
    that stopped; `text` the table the command prints."""
    outputs: tuple
    lines: tuple
    failed: tuple
    runs: int
    modes: dict
    text: str

    @property
    def stable(self) -> np.ndarray:
        return np.array([o.stable for o in self.outputs])

    @property
    def values(self) -> np.ndarray:
        return np.array([o.value for o in self.outputs])

    def __str__(self):
        return self.text

    def __repr__(self):
        lo = min((o.stable for o in self.outputs), default=None)
        return "DigitsReport(%d numbers, %d runs%s)" % (
            len(self.outputs), self.runs, ", fewest stable digits %.1f" % lo if lo is not None else "")


def digits(path, *, sets=None, orders=True, randoms=3, fma=True, mca="mca", samples=16, seed=1,
           attrib=8) -> DigitsReport:
    """Run `viv digits` on the .viv file at `path` (a repository-relative
    path works in a checkout). `sets` replaces top-level constants, as
    `viv run --set`. `mca` is "mca", "rr", "pb" or "off"; `attrib` the number
    of source lines perturbed one at a time (0 for none)."""
    p = str(_spec.source(path)) if not os.path.isabs(str(path)) and not os.path.exists(str(path)) else str(path)
    args = [_core.compiler(), "digits", "--randoms", str(randoms), "--mca", mca, "--samples", str(samples),
            "--seed", str(seed), "--attrib", str(attrib)]
    if not orders:
        args.append("--no-order")
    if not fma:
        args.append("--no-fma")
    for k, v in (sets or {}).items():
        v = ('"%s"' % v) if isinstance(v, str) else repr(int(v) if isinstance(v, (bool, np.integer)) else v)
        args += ["--set", "%s=%s" % (k, v)]
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        r = subprocess.run(args + ["-o", tmp, p], capture_output=True, text=True, env=_core._env())
        if r.returncode != 0:
            raise RuntimeError("viv digits %s: %s" % (path, r.stderr.strip()))
        with open(tmp, encoding="utf-8") as f:
            d = json.load(f)
    finally:
        os.unlink(tmp)
    outs = tuple(Output(o["value"], o["text"], o["order"], o["fma"], o["mca"], o["stable"]) for o in d["outputs"])
    return DigitsReport(outs, tuple(d["lines"]), tuple(d["failed"]), d["runs"], d["modes"], r.stdout)


def for_result(res, **opts) -> DigitsReport:
    """`Result.digits()`: the program behind a program result, run with
    the scalar parameters of that run."""
    spec = getattr(res, "_spec", None)
    if spec is None or spec.get("kind") != "program":
        raise TypeError("digits() needs the result of a vipyr program (statmech.ising, scf.run, ...)")
    sets = {k: v for k, v in (res.params or {}).items()
            if v is not None and isinstance(v, (int, float, str, np.integer, np.floating))}
    return digits(spec["sources"][0], sets=sets, **opts)
