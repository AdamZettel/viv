#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""check_rules_equiv.py: prove that the rule-driven py2viv translates as the translator it
replaced, whose NumPy and builtin mappings were Python code, except where a listed hole was
fixed.

    python3 tools/py2viv/check_rules_equiv.py                  (or check_rules_equiv.sh)
    python3 tools/py2viv/check_rules_equiv.py --base <git rev> --only examples,posts

The baseline is tools/py2viv/py2viv.py at BASE (default 252b401, the merge that brought
py2viv 0.2.0 to main, whose NumPy and builtin mappings were Python code), read with
`git show` and imported beside the current one. Every input is translated by both, in both
print modes (exact: print(... fmt(x) ...); plain), and these are compared:

  - the generated Viv, byte for byte (also for a file with refusals: its prelude, its
    `# REFUSED:` lines);
  - the inexact operations the manifest lists, and the run-time stops;
  - the refusal messages (file, line, column, caret, text, hint);
  - the manifest: print sites with each value's static type, the line map, refusals with
    their gaps and cascade flags;
  - whether the translator raised: a crash in the new one fails the proof.

Until 2026-09-27 they had to be identical (the rules were moved out of the code with the
output unchanged). Since then a difference is accepted only where one of FIXES (below, and
RULES.md, "Fixed holes") explains it: each differing Viv line is traced through the line map
to the Python statement it translates (or to the helper def it belongs to), each differing
message and manifest entry through its line, and that statement must use a name the fix
changed (np.transpose, .fill, an operator, an argument no parameter takes, ...), directly or
through a variable it reads; a helper must be one the fix changed, or one that such a helper
now needs or no longer needs. Temporaries are compared up to their numbering (a fix that
hoists one fewer renumbers the rest), print sites up to theirs. The report counts the
translations each fix changed; a difference no fix explains is printed and fails the proof.

Inputs:

  examples   tools/py2viv/examples/*.py
  posts      posts/refs/*.py, every file (those that import SciPy or torch are refused, and
             their refusals compared)
  ta         the TheAlgorithms units corpus.py builds (functions plus drivers); needs Python
             3.12 for PEP 695 syntax (check_rules_equiv.sh runs it under PY312), skipped
             under an older Python
  tests      every source tests/test_py2viv.py passes to the translator, recorded by running
             the suite with translate() wrapped (the Differential class is skipped: it
             translates examples/adapters.py, already an input)
  generated  programs written by this script, which call every rule's pattern with
             arguments of every static type, too few and too many arguments, known and
             unknown keywords, arguments that need hoisted statements (a conditional
             expression, a generator), arguments that are refused, and names that reach a
             rule through a `*` pattern (np.linalg.sqrt, np.random.rand); each call appears
             as an assignment, a print (so its type reaches the manifest) and a statement
"""

import argparse
import ast
import difflib
import importlib.util
import io
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import py2viv  # noqa: E402

BASE = "252b401"
MODES = ["exact", "plain"]


def load_base(rev):
    src = subprocess.run(["git", "-C", ROOT, "show", "%s:tools/py2viv/py2viv.py" % rev],
                         capture_output=True, text=True, check=True).stdout
    d = tempfile.mkdtemp(prefix="py2viv_base_")
    path = os.path.join(d, "py2viv_base.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("py2viv_base", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def outcome(mod, src, filename, print_mode):
    try:
        res = mod.translate(src, filename, print_mode=print_mode)
    except Exception as e:  # the baseline's crashes must be the new one's crashes
        return {"exception": type(e).__name__}
    try:
        man = json.dumps(mod.manifest(res, filename, src, print_mode), sort_keys=True)
    except Exception as e:  # manifest() of a syntax error fails in the baseline too
        man = "manifest raised " + type(e).__name__
    return {"viv": res["viv"], "messages": res["messages"], "manifest": man}


def first_diff(a, b):
    if "exception" in a and "exception" in b:
        return None  # both stop (a call Python rejects too, such as a missing argument)
    for k in ("exception", "messages", "viv", "manifest"):
        if a.get(k) != b.get(k):
            x, y = a.get(k), b.get(k)
            if isinstance(x, list) and isinstance(y, list):
                for i, (p, q) in enumerate(zip(x, y)):
                    if p != q:
                        return "%s, message %d:\n      base: %s\n      new:  %s" % (k, i + 1, p[:400], q[:400])
                return "%s: %d and %d messages" % (k, len(x), len(y))
            if isinstance(x, str) and isinstance(y, str):
                xl, yl = x.splitlines(), y.splitlines()
                for i, (p, q) in enumerate(zip(xl, yl)):
                    if p != q:
                        return "%s, line %d:\n      base: %s\n      new:  %s" % (k, i + 1, p[:200], q[:200])
                return "%s: lengths %d and %d lines" % (k, len(xl), len(yl))
            return "%s:\n      base: %s\n      new:  %s" % (k, str(x)[:300], str(y)[:300])
    return None


# ---------------------------------------------------------------- the listed fixes
#
# Since 2026-09-27 the translation differs from the baseline where a hole was fixed
# (RULES.md, "Fixed holes"). A difference is accepted only where it is explained: every Viv
# line, message and manifest entry that differs is traced to the Python statement it
# translates (the line map), or to the helper def it belongs to, and that statement must use a
# name one of these fixes changed (or the helper must be one a fix changed). A difference no
# fix explains fails the proof, as before.

_NP_TYPES = {"np.abs", "np.absolute", "np.diag", "np.diff", "np.cross", "np.outer", "np.cumsum", "np.prod", "np.trace",
             "np.max", "np.amax", "np.min", "np.amin", "np.square", "np.sign", "np.where", "np.clip", "np.maximum",
             "np.minimum", "np.fmax", "np.fmin", "np.full", "np.round", "np.around", "np.rint", "np.trunc", "np.fmod",
             "np.power", "math.fabs", "np.roll", "np.transpose", "np.real", "np.imag", "np.conj", "np.conjugate",
             "np.fabs", "np.floor", "np.ceil", "np.sqrt", "np.exp", "np.log", "np.sin", "np.cos", "np.tan",
             "np.tanh", "np.sinh", "np.cosh", "np.arcsin", "np.arccos", "np.arctan", "np.log10", "np.log2",
             "np.hypot", "np.arctan2", "np.sort", "sorted", ".sort", "np.copy", "list", "np.zeros_like", "np.empty_like",
             "np.ones_like", "np.full_like", "np.sum", "np.mean", "np.std", "np.var", "np.argmax", "np.argmin",
             ".cumsum", ".prod", ".max", ".min", ".copy", ".sum", ".dot", ".astype", ".argmax", ".argmin",
             ".mean", ".std", ".var", "np.array", "np.asarray", "np.float64", "float", "int", "math.trunc",
             "np.isnan", "np.isinf", "np.isfinite", "math.isnan", "math.isinf", "math.isfinite", "len",
             "np.dot", "np.matmul", "np.inner", "np.vdot", "np.linalg.norm", "np.isclose", "np.allclose"}
_MATH = {"math." + f for f in ("sqrt", "exp", "log", "sin", "cos", "tan", "tanh", "sinh", "cosh", "asin", "acos",
                               "atan", "erf", "gamma", "floor", "ceil", "fabs", "pow", "fmod", "isclose", "isnan",
                               "isinf", "isfinite", "hypot", "atan2", "log10", "log2", "trunc")}

FIXES = [
    ("precedence", "a template that is its argument alone kept precedence 100: np.transpose(v + 1.0) * 2.0 was "
     "v + 1.0 * 2.0 (also np.real, v.transpose(), round and math.floor/ceil of an int)",
     {"np.transpose", "np.real", ".transpose", "round", "math.floor", "math.ceil", "np.round", "np.around",
      "int", "math.trunc", "float", "np.float64", "np.conj", "np.conjugate", ".real"}, set(), set()),
    ("log-out", "np.log(x, b) was a logarithm to base b (NumPy's second argument is `out`); math.log(x, 1) "
     "divides by zero as CPython does", {"np.log", "math.log"}, {"py_div_f"}, set()),
    ("module-path", "a name with a module path in the middle (np.emath.sqrt, np.linalg.sqrt) or one no module "
     "has (math.abs, np.erf) reached the real elementwise function", {"<module-path>"}, set(), set()),
    ("binding", "an argument past the parameters, or a keyword the function does not take, was ignored "
     "(np.trace(A, 1), np.zeros(3, int) now binds its dtype); a missing argument crashed the translator",
     {"<binding>"}, set(), set()),
    ("complex-dot-norm", "np.vdot did not conjugate, np.dot of complex vectors was typed real, and "
     "np.linalg.norm of a complex array squared it; the products check their shapes",
     {"np.vdot", "np.dot", "np.matmul", "np.inner", ".dot", "np.linalg.norm"},
     {"py_cdot", "py_cvdot", "py_cnorm", "py_mm", "py_mv", "py_vm", "py_vdot", "py_vmul"}, set()),
    ("like-complex", "np.ones_like and np.full_like of a complex array gave czeros; np.full_like cast its value "
     "and lost a -0.0", {"np.ones_like", "np.full_like"}, {"py_cfillin", "py_fillin"}, set()),
    ("array-dtype", "np.array(a, dtype=int) of an existing array ignored the dtype; np.asarray of an array "
     "copied it (NumPy returns the same array); a dtype NumPy would round to (np.float32) was dropped",
     {"np.array", "np.asarray"}, set(), set()),
    ("sign-nan", "np.sign of an array gave 0 for NaN", {"np.sign"}, {"py_signs", "py_sign"}, set()),
    ("arctan2-type", "np.arctan2 of arrays was typed as a number", {"np.arctan2", "math.atan2"}, set(), set()),
    ("dtypes", "a result's type follows NumPy's dtype (an int array stays int, np.trunc and np.rint of an int "
     "are floats, np.full(3, 2) is int)", _NP_TYPES, {"py_int_arr", "py_int", "py_nptrunc"}, set()),
    ("sizes", "a negative size gave an empty array (NumPy raises ValueError); a float size was accepted",
     {"np.zeros", "np.empty", "np.ones", "np.full", "np.eye", "np.identity", "np.linspace", "np.arange"},
     {"py_size", "py_linspace", "py_arange"}, set()),
    ("math-scalars", "a math function of an array or a Bool (TypeError, or float16 in NumPy) was translated",
     _MATH | {"abs"}, set(), set()),
    ("builtins", "the builtins sum, min, max, all, any, list, abs, float, int and round of a matrix, a list or an "
     "array (Python adds rows, raises TypeError, or copies shallowly) were translated",
     {"sum", "min", "max", "all", "any", "list", "abs", "float", "int", "round", "len", "sorted", "np.all", "np.any"},
     set(), set()),
    ("hypot", "hypot as sqrt(x*x + y*y) overflowed and underflowed where hypot does not", {"math.hypot", "np.hypot"},
     {"py_hypot", "py_hypot_vv", "py_hypot_vs"}, set()),
    ("zero-signs", "np.maximum, np.minimum, np.fmax, np.fmin, np.max, np.min and np.clip used Viv's max and min, "
     "whose result for two zeros differs between engines; np.clip propagated a NaN bound (NumPy ignores it); "
     "np.cumsum started from +0.0",
     {"np.maximum", "np.minimum", "np.fmax", "np.fmin", "np.max", "np.amax", "np.min", "np.amin", ".max", ".min",
      "np.clip", "np.cumsum", ".cumsum"},
     {"py_npmax2", "py_npmin2", "py_npmaxs", "py_npmins", "py_npmaxv", "py_npminv", "py_npmax", "py_npmin", "py_zmax2",
      "py_zmin2", "py_fmax2", "py_fmin2", "py_fmaxv", "py_fminv", "py_fmaxs", "py_fmins", "py_clip", "py_clipv",
      "py_cumsum"}, set()),
    ("linalg", "np.linalg.cholesky and eigvalsh read the whole matrix (LAPACK reads the lower triangle); a "
     "singular or non-finite input gave numbers where NumPy raises or LAPACK's answer is its own",
     {"np.linalg.solve", "np.linalg.inv", "np.linalg.cholesky", "np.linalg.eigvalsh", "np.fft.fft", "np.fft.ifft",
      "<np.linalg>", "<np.fft>"},
     {"py_solve", "py_solve_multi", "py_inv", "py_cholesky", "py_eigvalsh", "py_lower_sym", "py_fft_r", "py_fft_c",
      "py_ifft_r", "py_ifft_c"}, set()),
    ("views", "a transpose, or the real or imaginary part of a complex array, is a view in NumPy and a copy in "
     "Viv: a name bound to one may not be written, nor the array after it", {".T", ".transpose", "np.transpose",
                                                                             "np.real", "np.imag", ".real", ".imag"},
     set(), set()),
    ("double-eval", "a template that writes an argument twice evaluated a def call in it twice",
     {"<twice>"}, set(), set()),
    ("operators", "Viv types `-a` and `a ^ y` of an array a def returned as a number (now -1.0 * a and py_apow); "
     "arithmetic on a Python list, a shape, two Bool arrays, or a real array with a complex number, and "
     "negating a list, a Bool array or a complex array, are refused", {"<unary>", "<pow>", "<binop>", "pow", "divmod"},
     {"py_apow"}, set()),
    ("lists", "a method or attribute of a Python list (AttributeError) was translated", {"<list-member>"}, set(),
     set()),
    ("isclose-kw", "np.isclose and np.allclose ignored an unknown keyword; math.isclose read its keywords by "
     "name", {"np.isclose", "np.allclose", "math.isclose"}, set(), set()),
    ("cross", "np.cross of vectors that are not 3-vectors gave a 3-vector (NumPy gives a number for 2-vectors)",
     {"np.cross"}, {"py_cross"}, set()),
    ("erf-gamma", "math.erf(inf) and math.gamma(inf) were NaN (Viv's vmath)", {"math.erf", "math.gamma"},
     {"py_merf", "py_mgamma"}, set()),
    ("hoist-order", "a hoisted expression that calls a def ran before a call of a def Python evaluates first",
     {"<hoist>"}, set(), set()),
    ("fill", "a.fill(v) of a number or a string (AttributeError in Python), of an int array with a float (NumPy "
     "truncates), of a Bool array with a number, or with a complex value was translated", {".fill"}, set(), set()),
    ("fixed-crash", "a call with a missing argument crashed the translator, and so did --manifest of a file "
     "with a syntax error", {"<crash>"}, set(), set()),
    # the NumPy core (branch np-core): new translations and the holes found on the way
    ("slices", "a subscript with a range translates (it was refused, R3), a store into a slice or a row too; "
     "the view check (views.py) refuses a slice, row or transpose a write could show", {"<slice>", "<subscript>", "<for>"},
     {"py_lo", "py_hi", "py_ix", "py_fit", "py_cfit", "py_b01"}, set()),
    ("npsum-blocks", "np.sum of more than 8192 floats was one pairwise sum (NumPy sums 8192 at a time), and of 8 "
     "or more -0.0 gave -0.0", {"np.sum", ".sum", "np.mean", ".mean", "np.std", ".std", "np.var", ".var", "np.prod", ".prod", "np.max", "np.amax",
      ".max", "np.min", "np.amin", ".min", "np.argmax", ".argmax", "np.argmin", ".argmin", "np.cumsum", ".cumsum", "np.trace"}, {"py_npsum", "py_blk", "py_pw", "py_mean", "py_var", "py_std"}, set()),
    ("axis", "reductions along an axis and with keepdims translate (they were refused, R9)", {"np.sum", ".sum", "np.mean", ".mean", "np.std", ".std", "np.var", ".var", "np.prod", ".prod", "np.max", "np.amax",
      ".max", "np.min", "np.amin", ".min", "np.argmax", ".argmax", "np.argmin", ".argmin", "np.cumsum", ".cumsum", "np.trace"},
     {"py_ax", "py_cax", "py_rdims", "py_shaped", "py_sum_fl", "py_sum_ax", "py_isum_ax", "py_mean_ax", "py_var_fl",
      "py_var_ax", "py_std_ax", "py_prod_ax", "py_max_ax", "py_argmax_ax", "py_cumsum_ax", "py_iota", "py_flat",
      "py_zmax2", "py_zmin2", "py_int_arr", "py_int", "py_cumsum"}, set()),
    ("transpose-order", "a sum, mean, variance or product of a transpose summed its row-order copy (NumPy sums in "
     "memory order)", {".T", "np.transpose", ".transpose"} | {"np.sum", ".sum", "np.mean", ".mean", "np.std", ".std", "np.var", ".var", "np.prod", ".prod", "np.max", "np.amax",
      ".max", "np.min", "np.amin", ".min", "np.argmax", ".argmax", "np.argmin", ".argmin", "np.cumsum", ".cumsum", "np.trace"}, set(), set()),
    ("augassign-alias", "`x += v` of an array another name or a caller holds rebound x in Viv (NumPy changes the "
     "array in place)", {"<augassign>"}, {"py_fit", "py_cfit"}, set()),
    ("reshape-joins", "np.reshape, a.reshape, ravel, flatten, concatenate, the stacks and meshgrid translate (they "
     "were refused, R10)", {"np.reshape", ".reshape", "np.ravel", ".ravel", ".flatten", "np.concatenate", "np.stack",
                            "np.vstack", "np.row_stack", "np.hstack", "np.column_stack", "np.meshgrid", "np.flatten"},
     {"py_reshape", "py_shaped", "py_meshgrid", "py_iota", "py_flat"}, set()),
    ("rank-3-literals", "a rank-3 list literal translated, and the C backend does not build it", {"<literal3>", "np.array"},
     set(), set()),
    ("astype-keyword", "a.astype's parameter is `dtype`, as NumPy names it", {".astype"}, set(), set()),
    # np.random (branch np-random): new translations and the holes found on the way
    ("np-random", "np.random's legacy functions after np.random.seed, and a Generator's methods, translate draw "
     "for draw through lib/nprandom (they were refused); the refusals of the rest name the reason",
     {"<np.random>", ".random", ".integers", ".uniform", ".standard_normal", ".normal", ".choice", ".shuffle",
      ".permutation", "<import numpy.random>"},
     {"npr_bits", "npr_mt19937", "npr_legacy", "npr_seedseq", "npr_pcg64", "npr_zig", "npr_generator", "py_np_mt",
      "py_npr_signbit", "py_npr_st", "py_npr_seed", "py_npr_seed_array", "py_npr_rand", "py_npr_uniform",
      "py_npr_randint", "py_npr_shuffle", "py_npr_permutation", "py_npr_pcheck", "py_npr_choice", "py_npg_new",
      "py_npg_random", "py_npg_normal", "py_npg_uniform", "py_npg_integers", "py_npg_shuffle", "py_npg_choice",
      "py_b01", "py_size"}, set()),
    ("negative-index-eval", "f(x)[-1] wrote f(x) twice (its length), so a def with an effect ran twice",
     {"<neg-index>"}, set(), set()),
    # growable lists (branch growable, R17): the fix was never listed here
    ("growable", "`lst = []` grown by `lst.append(x)` translates (R17); the other list methods and an empty list "
     "elsewhere are refused with R17's messages", {".append", ".extend", ".insert", ".pop", ".remove", ".index",
                                                    ".count", ".clear", "<emptylist>", "<list-member>", "<for>"},
     set(), set()),
    # broadcasting, masks and integer indices (branch np-bcast)
    ("broadcast", "operators of two arrays broadcast as NumPy does (two shapes of one rank with a length 1 stopped "
     "Viv; two ranks were refused, R9); `x += v` of an array writes into x (py_inpl); a store into a slice "
     "broadcasts (py_bset); np.maximum, np.minimum, np.fmax, np.fmin, np.where, np.arctan2, np.hypot and np.allclose "
     "broadcast", {"<binop>", "<compare>", "<augassign>", "<subscript>", "<slice>", "np.maximum", "np.minimum", "np.fmax",
                   "np.fmin", "np.where", "np.arctan2", "math.atan2", "np.hypot", "np.allclose", "np.isclose"},
     {"py_badd", "py_ballclose", "py_batan2", "py_bdims", "py_bdimsq", "py_bdiv", "py_beq", "py_bfits", "py_bge",
      "py_bgt", "py_bhypot", "py_bidx", "py_ble", "py_blt", "py_bmul", "py_bne", "py_bset", "py_bsub", "py_bto",
      "py_bwhere", "py_cbadd", "py_cbmul", "py_cbmulcr", "py_cbmulrc", "py_cbset", "py_cbsub", "py_cbto", "py_cfit", "py_cinpl",
      "py_cshaped", "py_fit", "py_fmaxb", "py_fminb", "py_inpl", "py_npmaxb", "py_npminb"}, set()),
    ("masks-fancy", "a mask or an integer array among a subscript's items, `None` in one, np.nonzero(a)[k], "
     "np.where(m)[k], np.flatnonzero, np.count_nonzero and np.argwhere translate (they were refused)",
     {"<subscript>", "<slice>", "<augassign>", "<binop>", "np.nonzero", "np.where", "np.flatnonzero", "np.count_nonzero",
      "np.argwhere", "np.newaxis"},
     {"py_argwhere", "py_bdimsi", "py_gath", "py_gath2", "py_gath3", "py_ixchk", "py_mask", "py_maskax", "py_mchk",
      "py_mset", "py_msetax", "py_newax", "py_nz2", "py_nzax", "py_nzflat", "py_set2", "py_set3", "py_setax",
      "py_vals"}, set()),
    ("none-index", "`a[None]` read as `a[0]`, and a store through it wrote a[0]", {"<subscript>"}, set(), set()),
    ("complex-ops", "`/` of two arrays one of which is complex did not build; a comparison of complex arrays gave one "
     "Bool", {"<binop>", "<compare>"}, set(), set()),
    ("list-compare", "a comparison of two Python lists, or of a list and a number, gave an array", {"<compare>"}, set(),
     set()),
    ("allclose-shapes", "np.allclose compared two arrays of one size and different shapes flat", {"np.allclose"},
     {"py_ballclose", "py_allclose"}, set()),
    ("augassign-dtype", "`x += 0.5` and `x /= 2` of an integer array no other name holds rebound x as a float array",
     {"<augassign>"}, set(), set()),
    ("element-array", "an array stored into one element translated (and did not build)", {"<subscript>"}, set(), set()),
]


_TMP = re.compile(r"\bpy_([a-z]{1,5})(\d+)\b")


def _untmp(text):
    """Temporaries renumbered: a fix that hoists one temporary fewer renumbers the rest."""
    helpers = py2viv.RULES.helpers
    return _TMP.sub(lambda m: m.group(0) if m.group(0) in helpers else "py_%s#" % m.group(1), text)


def _helper_blocks(lines):
    """name -> the lines of each helper def in a translation's prelude (its #| comment too)."""
    out = {}
    cur = None
    buf = []
    for l in lines:
        m = re.match(r"def (\w+)\(", l)
        if l.startswith("#|") and cur is None:
            buf.append(l)
            continue
        if m:
            cur = m.group(1)
            buf.append(l)
            continue
        if cur is not None:
            buf.append(l)
            if l == "}":
                out[cur] = "\n".join(buf)
                cur = None
                buf = []
            continue
        buf = []
    return out


class _Names(ast.NodeVisitor):
    """The names a Python statement uses, as fix names: calls (np.linalg.solve, abs), methods
    and attributes (.sum, .T), and markers for operators and argument lists."""

    def __init__(self, aliases, defs=(), lists=()):
        self.aliases = aliases
        self.defs = set(defs)    # the program's defs (a call of one may have effects)
        self.lists = set(lists)  # names bound to a list literal somewhere in the program
        self.out = set()

    def dotted(self, f):
        parts = []
        while isinstance(f, ast.Attribute):
            parts.append(f.attr)
            f = f.value
        if not isinstance(f, ast.Name):
            return None
        root = self.aliases.get(f.id, f.id)
        return ".".join([root] + list(reversed(parts)))

    def visit_Call(self, n):
        name = self.dotted(n.func)
        if name is not None and (name.split(".")[0] in ("np", "math") or "." not in name):
            self.out.add(name)
            parts = name.split(".")
            if parts[0] == "np" and len(parts) >= 3 and parts[1] not in ("linalg", "fft", "random"):
                self.out.add("<module-path>")
            if parts[0] == "np" and len(parts) >= 3 and parts[1] == "random":
                self.out.add("<np.random>")
            if parts[0] == "np" and len(parts) >= 3 and parts[1] in ("linalg", "fft"):
                self.out.add("<np.%s>" % parts[1])
                if parts[2] not in ("solve", "inv", "cholesky", "eigvalsh", "eigh", "norm", "fft", "ifft", "det",
                                    "eig", "svd", "qr", "lstsq", "matrix_rank", "pinv"):
                    self.out.add("<module-path>")
            if parts[0] == "math" and len(parts) == 2 and not hasattr(__import__("math"), parts[1]):
                self.out.add("<module-path>")
            if parts[0] == "np" and len(parts) == 2 and parts[1] in ("erf", "gamma", "asin", "acos", "atan"):
                self.out.add("<module-path>")
            if _binding_problem(name, n):
                self.out.add("<binding>")
        elif isinstance(n.func, ast.Attribute):
            self.out.add("." + n.func.attr)
            if _binding_problem(n.func.attr, n, method=True):
                self.out.add("<binding>")
            if isinstance(n.func.value, ast.Name) and n.func.value.id in self.lists:
                self.out.add("<list-member>")
        if any(isinstance(m, ast.Call) and isinstance(m.func, ast.Name) and m.func.id in self.defs
               for a in n.args for m in ast.walk(a)):
            self.out.add("<twice>")
        self.generic_visit(n)

    def visit_Attribute(self, n):
        name = self.dotted(n)
        if name is None or name.split(".")[0] not in ("np", "math"):
            self.out.add("." + n.attr)
            if isinstance(n.value, ast.Name) and n.value.id in self.lists:
                self.out.add("<list-member>")
        self.generic_visit(n)

    def visit_Import(self, n):
        if any(a.name == "numpy.random" for a in n.names):
            self.out.add("<import numpy.random>")
        self.generic_visit(n)

    def visit_ImportFrom(self, n):
        if n.module == "numpy.random":
            self.out.add("<import numpy.random>")
        self.generic_visit(n)

    def visit_IfExp(self, n):
        self.out.add("<hoist>")
        self.generic_visit(n)

    def visit_ListComp(self, n):
        self.out.add("<hoist>")
        self.generic_visit(n)

    def visit_GeneratorExp(self, n):
        self.out.add("<hoist>")
        self.generic_visit(n)

    def visit_UnaryOp(self, n):
        self.out.add("<unary>")
        self.generic_visit(n)

    def visit_BinOp(self, n):
        self.out.add("<pow>" if isinstance(n.op, ast.Pow) else "<binop>")
        self.generic_visit(n)

    def visit_AugAssign(self, n):
        self.out.add("<pow>" if isinstance(n.op, ast.Pow) else "<binop>")
        if isinstance(n.target, ast.Name):
            self.out.add("<augassign>")
        self.generic_visit(n)

    def visit_Compare(self, n):
        self.out.add("<compare>")
        self.generic_visit(n)

    def visit_Subscript(self, n):
        sl = n.slice
        items = sl.elts if isinstance(sl, ast.Tuple) else [sl]
        if any(isinstance(i, ast.Slice) for i in items):
            self.out.add("<slice>")
        self.out.add("<subscript>")
        if not isinstance(n.value, ast.Name) and any(isinstance(i, ast.UnaryOp) and isinstance(i.op, ast.USub)
                                                     and isinstance(i.operand, ast.Constant) for i in items):
            self.out.add("<neg-index>")
        self.generic_visit(n)

    def visit_List(self, n):
        if not n.elts or any(isinstance(t, ast.Name) and t.id in self.lists for t in ast.walk(n)):
            self.out.add("<emptylist>")
        depth, x = 0, n
        while isinstance(x, (ast.List, ast.Tuple)) and x.elts:
            depth += 1
            x = x.elts[0]
        if depth >= 3:
            self.out.add("<literal3>")
        self.generic_visit(n)

    def visit_For(self, n):
        self.out.add("<for>")
        self.generic_visit(n)


def _binding_problem(name, call, method=False):
    """The call passes an argument no pattern of the name takes, or misses one it needs."""
    kind = "method" if method else "call"
    cands = py2viv.RULES.candidates(kind, name)
    if not cands:
        return False
    if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
        return True
    for pat, rule in cands:
        if pat.anyargs:
            return False
        pos = [p for p in pat.params if p[1] in ("pos", "poskw")]
        kwok = {p[0] for p in pat.params if p[1] in ("poskw", "kw")}
        if len(call.args) > len(pos) and not pat.varargs:
            continue
        if any(k.arg not in kwok for k in call.keywords) and not pat.varkw:
            continue
        given = {p[0] for p in pos[:len(call.args)]} | {k.arg for k in call.keywords}
        if any(not p[3] and p[0] not in given for p in pat.params):
            continue
        if any(i < len(pos) and pos[i][1] != "pos" and pos[i][0] in {k.arg for k in call.keywords}
               for i in range(len(call.args))):
            continue
        return False
    return True


def _aliases(tree):
    al = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name in ("numpy", "math"):
                    al[a.asname or a.name] = "np" if a.name == "numpy" else "math"
        elif isinstance(n, ast.ImportFrom) and n.module in ("numpy", "math", "numpy.linalg", "numpy.fft", "numpy.random"):
            root = "np" if n.module.startswith("numpy") else "math"
            sub = n.module.split(".")[1:] if n.module.startswith("numpy.") else []
            for a in n.names:
                al[a.asname or a.name] = ".".join([root] + sub + [a.name])
    return al


def _statement_at(tree, line):
    """The innermost simple statement on a Python line (for a compound statement, its header)."""
    best = None
    for n in ast.walk(tree):
        if isinstance(n, ast.stmt) and n.lineno <= line <= getattr(n, "end_lineno", n.lineno):
            if best is None or (n.lineno >= best.lineno and getattr(n, "end_lineno", 0) <= getattr(best, "end_lineno", 0)):
                best = n
    if best is None:
        return None
    if isinstance(best, (ast.If, ast.While)):
        return best.test
    if isinstance(best, ast.For):
        return best.iter
    if isinstance(best, (ast.FunctionDef,)):
        return None
    return best


def _names_at(src, line, deep=False, calls_too=True):
    """The fix names a Python line's statement uses; with deep, also those of every statement
    that assigns a variable it reads (a type a fix changed reaches a later print or call)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {"<crash>"}
    node = _statement_at(tree, line)
    if node is None:
        # a def's own line: its parameters are typed from its calls, so a fix that changed the
        # type of an argument changed it
        for f in ast.walk(tree):
            if isinstance(f, ast.FunctionDef) and f.lineno == line:
                out = set()
                for c in ast.walk(tree):
                    if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == f.name:
                        out |= _names_at(src, c.lineno)
                return out
        return set()
    if deep:
        reads = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        out = set(_names_at(src, line))
        # the headers of the compound statements around it (a body whose `if` is now refused)
        for st in ast.walk(tree):
            if isinstance(st, (ast.If, ast.While, ast.For)) and st.lineno < line <= getattr(st, "end_lineno", st.lineno):
                out |= _names_at(src, st.lineno)
        for st in ast.walk(tree):
            if isinstance(st, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                tgts = st.targets if isinstance(st, ast.Assign) else [st.target]
                if any(isinstance(t, ast.Name) and t.id in reads for tg in tgts for t in ast.walk(tg)):
                    out |= _names_at(src, st.lineno)
        return out
    defs = {f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)}
    lists = {t.id for m in ast.walk(tree) if isinstance(m, ast.Assign) and isinstance(m.value, (ast.List, ast.ListComp))
             for t in m.targets if isinstance(t, ast.Name)}
    v = _Names(_aliases(tree), defs, lists)
    v.visit(node)
    if not calls_too:
        return v.out
    # a def's parameters are typed from all its calls: a call here is refused ("`f` called with a
    # number ..., and elsewhere with an array") when a fix changed the argument of another call
    out = set(v.out)
    called = {c.func.id for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in defs}
    for c in ast.walk(tree):
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in called:
            out |= _names_at(src, c.lineno, calls_too=False)
    return out


def _helper_of(viv_lines, i):
    """The helper def a prelude line belongs to (None for the program's own lines)."""
    h = _helper_of_any(viv_lines, i)
    return h if h in py2viv.RULES.helpers else None


def _helper_of_any(viv_lines, i):
    """The def a line belongs to: its `#|` comment runs up to the def, its body down from it."""
    line = viv_lines[i]
    if line.startswith("#|") or line.startswith("def "):
        for j in range(i, len(viv_lines)):
            m = re.match(r"def (\w+)\(", viv_lines[j])
            if m:
                return m.group(1)
            if not viv_lines[j].startswith("#|"):
                break
        return None
    for j in range(i, -1, -1):
        m = re.match(r"def (\w+)\(", viv_lines[j])
        if m:
            return m.group(1)
        if viv_lines[j] == "" or not (viv_lines[j].startswith(" ") or viv_lines[j].startswith("}")):
            break
    return None


def _rule_names_using(helper):
    """The names of the rules whose cases use this helper."""
    out = set()
    for r in py2viv.RULES.rules:
        if any(helper in c.get("use", []) or any(helper == u.replace("{short}", p.key.split(".")[-1]).replace("{name}", p.key)
                                                 for u in c.get("use", []) for p in r.patterns)
               for c in r.cases):
            for p in r.patterns:
                out.add(("." + p.key) if p.receiver else p.key)
    return out


def _fixes_for(names, helper=None):
    out = []
    for fid, desc, fnames, fhelpers, _ in FIXES:
        if names & fnames or (helper is not None and helper in fhelpers):
            out.append(fid)
    return out


def explain(a, b, src):
    """The fixes that explain every difference between two outcomes, or (None, where) for the
    first difference none explains."""
    if "exception" in b:
        return None, "the new translator crashes: %s" % b["exception"]
    if "exception" in a:
        return ["fixed-crash"], None
    used = set()
    follows = []
    lines_new = _untmp(b["viv"]).splitlines()
    lines_old = _untmp(a["viv"]).splitlines()
    blocks_new = _helper_blocks(lines_new)
    blocks_old = _helper_blocks(lines_old)
    man_a = json.loads(a["manifest"]) if not a["manifest"].startswith("manifest raised") else None
    man_b = json.loads(b["manifest"])
    lm_b = man_b["linemap"]
    lm_a = man_a["linemap"] if man_a else []
    sm = difflib.SequenceMatcher(None, lines_old, lines_new, autojunk=False)

    def attribute(pyline, helper, where):
        if pyline:
            f = _fixes_for(_names_at(src, pyline)) or _fixes_for(_names_at(src, pyline, deep=True))
        else:
            f = _fixes_for(set(), helper) if helper else []
            if not f and helper and (helper in blocks_new) != (helper in blocks_old):
                # a helper carried or dropped: the fixes of the rules that use it
                f = _fixes_for(_rule_names_using(helper))
            if not f and helper and blocks_new.get(helper) == blocks_old.get(helper) or \
                    not f and helper and (helper in blocks_new) != (helper in blocks_old) and \
                    helper in py2viv.RULES.helpers:
                # a helper whose text is unchanged, carried or dropped because a helper a fix
                # changed now needs it or no longer does: explained if something else is
                follows.append(where)
                return None
        if not f:
            return where
        used.update(f)
        return None

    def pyline_of(lm, k):
        """The Python line of Viv line k: its own, or (a closing brace, a hoisted line) the
        nearest one above it in the program."""
        pl = lm[k] if k < len(lm) else 0
        j = k
        while not pl and j > 0 and _helper_of_lines(lm, k) is None:
            j -= 1
            pl = lm[j] if j < len(lm) else 0
        return pl

    def _helper_of_lines(lm, k):
        return None
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        for j in range(j1, j2):
            h = _helper_of(lines_new, j)
            pl = pyline_of(lm_b, j) if h is None else 0
            bad = attribute(pl, h if not pl else None,
                            "viv line %d (%s)" % (j + 1, lines_new[j][:120]))
            if bad and not lines_new[j].strip() == "":
                return None, bad
        for i in range(i1, i2):
            h = _helper_of(lines_old, i)
            pl = pyline_of(lm_a, i) if h is None else 0
            bad = attribute(pl, h if not pl else None,
                            "baseline viv line %d (%s)" % (i + 1, lines_old[i][:120]))
            if bad and not lines_old[i].strip() == "":
                return None, bad
    for m in set(a["messages"]) ^ set(b["messages"]):
        mm = re.match(r"[^:\n]*:(\d+):\d+:", m)
        bad = attribute(int(mm.group(1)) if mm else 0, None, "message: %s" % m.splitlines()[0][:160])
        if bad:
            return None, bad
    if man_a is None:
        used.add("fixed-crash")
    else:
        for key in ("sites", "inexact", "stops", "refusals"):
            # a print site's number counts the sites before it: a refused one renumbers the rest
            drop = (lambda e: {k: v for k, v in e.items() if k != "site"}) if key == "sites" else (lambda e: e)
            xa = {json.dumps(drop(e), sort_keys=True): e for e in man_a.get(key, [])}
            xb = {json.dumps(drop(e), sort_keys=True): e for e in man_b.get(key, [])}
            for k in set(xa) ^ set(xb):
                e = xa.get(k) or xb.get(k)
                # a helper's stop that states the start of NumPy's message: the helper's fix
                h = next((n for n, ms in py2viv.RULES.helper_stops.items() if e.get("message") in ms), None) \
                    if key == "stops" and e.get("prefix") else None
                bad = attribute(e.get("line", 0) if h is None else 0, h, "manifest %s: %s" % (key, k[:160]))
                if bad:
                    return None, bad
    if follows and not used:
        return None, follows[0]
    return sorted(used), None


# ---------------------------------------------------------------- inputs


def inputs_examples():
    d = os.path.join(HERE, "examples")
    for f in sorted(os.listdir(d)):
        if f.endswith(".py"):
            p = os.path.join(d, f)
            yield "examples/" + f, open(p).read(), p


def inputs_posts():
    d = os.path.join(ROOT, "posts", "refs")
    if not os.path.isdir(d):
        return
    for f in sorted(os.listdir(d)):
        if f.endswith(".py"):
            p = os.path.join(d, f)
            yield "posts/refs/" + f, open(p).read(), p


def inputs_ta():
    if sys.version_info < (3, 12):
        return
    import corpus
    src_dir = corpus.ta_src()
    if not src_dir:
        return
    for part, files in corpus.PORTED.items():
        for rel, names in files.items():
            src = corpus.fix_314(open(os.path.join(src_dir, rel)).read())
            for name in names:
                drv = corpus.DRIVERS.get(name)
                called = []
                if drv:
                    called = [m.func.id for m in ast.walk(ast.parse(drv)) if isinstance(m, ast.Call) and isinstance(m.func, ast.Name)]
                unit_src, _ = corpus.extract(src, [name] + [c for c in called if c != name])
                prog = unit_src + ("\n\n" + drv.strip() + "\n" if drv else "")
                uname = "%s__%s" % (rel.replace("/", "_")[:-3], name)
                yield "ta/" + uname, prog, uname + ".py"


def inputs_tests():
    """Every source the unit tests translate, recorded by running them."""
    seen = []
    real = py2viv.translate

    def recording(src, filename="<python>", **kw):
        seen.append((src, filename))
        return real(src, filename, **kw)
    sys.path.insert(0, os.path.join(HERE, "tests"))
    import test_py2viv
    py2viv.translate = recording
    test_py2viv.py2viv.translate = recording
    try:
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for cls in [c for c in vars(test_py2viv).values() if isinstance(c, type) and issubclass(c, unittest.TestCase)]:
            if cls.__name__ == "Differential":
                continue
            suite.addTests(loader.loadTestsFromTestCase(cls))
        unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    finally:
        py2viv.translate = real
        test_py2viv.py2viv.translate = real
    uniq = []
    for s in seen:
        if s not in uniq:
            uniq.append(s)
    for i, (src, fn) in enumerate(uniq):
        yield "tests/%03d" % i, src, fn


# ---------------------------------------------------------------- generated programs

PRELUDE = """\
import numpy as np
import math
from numpy import sqrt as np_sqrt, pi as np_pi
from math import floor as m_floor, tau
import scipy
n = 3
k = -2
f = 2.5
g = -0.0
b = True
v = np.array([1.0, -2.0, 3.5])
vi = np.array([1, 2, 3])
vb = v > 0.0
L = [1.0, 2.0, 4.0]
Li = [1, 2]
Lb = [True, False]
M = np.eye(3)
Mi = np.zeros((2, 2), dtype=int)
T3 = np.zeros((2, 2, 2))
X = np.zeros(4, dtype=complex)
z = 1.5 + 2j
H = np.array([[2.0, 1.0], [1.0, 3.0]])
Lm = [1, 2.5]
"""

ARGS = [
    "n", "k", "f", "g", "b", "v", "vi", "vb", "L", "Li", "Lb", "M", "Mi", "T3", "X", "z", "H",
    "v + 1.0", "f * 2", "-f", "f if b else 1.0", "n > 1", "[1.0, 2.0]", "[1, 2, 3]", "(n, n)",
    "(2, 3, 4)", "(1, 1, 1, 1, 1)", "[[1.0, 2.0], [3.0, 4.0]]", "[x * 2.0 for x in v]",
    "[x for x in range(n)]", "np.zeros(n)", "v.shape", "M.shape", "undefined_name",
    "scipy.pi", "{1: 2}", "'inf'", "'-Infinity'", "' nan '", "'abc'", "3j", "None", "float",
    "int", "complex", "np.float64", "np.int32", "np.complex128", "np.bool_", "bool", "1e300",
    "sum(x for x in v)", "[1.0 + 2j, 3j]", "Lm", "True", "-0.0", "0", "7",
]
GEN_ARGS = ["x * x for x in v", "x > 0 for x in v", "i for i in range(n) if i > 1"]
KEYWORDS = ["axis", "dtype", "k", "fill_value", "rtol", "atol", "rel_tol", "abs_tol", "num",
            "start", "stop", "equal_nan", "key", "default", "ddof", "out", "x", "a", "b", "foo"]
RECEIVERS = ["v", "vi", "M", "Mi", "L", "X", "T3", "f", "z", "(v + 1.0)", "undefined_name", "'abc'",
             "n", "b", "vb", "np_sqrt"]
METHODS = ["sum", "max", "min", "mean", "std", "var", "copy", "dot", "transpose", "astype", "tolist",
           "argmax", "argmin", "prod", "cumsum", "reshape", "flatten", "ravel", "squeeze", "append",
           "extend", "insert", "pop", "remove", "index", "count", "clear", "sort", "fill", "item",
           "conj"]
ATTRS = ["T", "shape", "size", "ndim", "real", "imag", "dtype", "flat"]
EXTRA_NAMES = ["np.linalg.sqrt", "np.emath.log", "np.fft.exp", "math.abs", "math.arcsin",
               "np.random.rand", "np.random.normal", "random.random", "np.fabs", "np.linalg.abs",
               "np.linalg.floor", "math.foo", "np.foo", "foo", "np.linalg.foo", "np_sqrt", "m_floor",
               "np.asin", "math.absolute", "np.linalg.cond"]


def rule_names():
    names = []
    for r in py2viv.RULES.rules:
        for p in r.patterns:
            if p.kind == "call" and not p.glob:
                names.append(p.key)
    return sorted(set(names)) + EXTRA_NAMES


def gen_call(rnd, name):
    nargs = rnd.choice([0, 1, 1, 1, 2, 2, 2, 3, 3, 4])
    args = [rnd.choice(ARGS) for _ in range(nargs)]
    if nargs == 1 and rnd.random() < 0.08:
        args = [rnd.choice(GEN_ARGS)]
    kws = []
    if rnd.random() < 0.3:
        for kw in rnd.sample(KEYWORDS, rnd.choice([1, 1, 2])):
            kws.append("%s=%s" % (kw, rnd.choice(ARGS[:30] + ["float", "int", "complex", "np.float32", "'float'"])))
    if rnd.random() < 0.02:
        kws.append("**{}")
    if rnd.random() < 0.02:
        args.append("*v")
    return "%s(%s)" % (name, ", ".join(args + kws))


def gen_method(rnd):
    r = rnd.choice(RECEIVERS)
    m = rnd.choice(METHODS)
    nargs = rnd.choice([0, 0, 0, 1, 1, 2])
    args = [rnd.choice(ARGS) for _ in range(nargs)]
    if rnd.random() < 0.08:
        args.append("axis=0")
    return "%s.%s(%s)" % (r, m, ", ".join(args))


def gen_attr(rnd):
    return "%s.%s" % (rnd.choice(RECEIVERS), rnd.choice(ATTRS))


CONSTS = ["np.pi", "math.pi", "np.e", "math.e", "math.tau", "np.inf", "math.inf", "np.Inf", "np.nan",
          "math.nan", "np.NaN", "np.random", "np.random.seed", "np.newaxis", "math.foo", "np_pi", "tau",
          "np.linalg", "np.float64"]


def gen_program(rnd, names, size=40):
    lines = []
    for i in range(size):
        roll = rnd.random()
        if roll < 0.62:
            e = gen_call(rnd, rnd.choice(names))
        elif roll < 0.82:
            e = gen_method(rnd)
        elif roll < 0.92:
            e = gen_attr(rnd)
        else:
            e = rnd.choice(CONSTS)
        form = rnd.random()
        if form < 0.45:
            lines.append("r%d = %s" % (i, e))
        elif form < 0.8:
            lines.append("print(%s)" % e)
        elif form < 0.9:
            lines.append(e)
        elif form < 0.95:
            lines.append("if %s:\n    r%d = 1.0" % (e, i))
        else:
            lines.append("r%d = %s * 2.0 - 1" % (i, e))
    if rnd.random() < 0.5:
        lines.append(rnd.choice(["w, V = np.linalg.eigh(H)", "w, V = np.linalg.eigh(M, 1)", "w = np.linalg.eigh(H)",
                                 "a, b, c = np.linalg.eigh(H)", "w, V[0] = np.linalg.eigh(H)",
                                 "w, V = np.linalg.eigh(H + 1.0)", "w, V = np.linalg.eigh(f if b else H)",
                                 "w, V = np.linalg.eigh(undefined_name)", "q, r = np.linalg.qr(H)"]))
        lines.append("print(w, V)")
    return lines


def gen_rule_programs(names):
    """One program per call name, calling it with a fixed spread of argument lists (so every
    rule is exercised with every argument kind, not only when the random draw picks it)."""
    for name in names:
        lines = []
        i = 0
        for a in ARGS:
            lines.append("r%d = %s(%s)" % (i, name, a))
            lines.append("print(%s(%s))" % (name, a))
            i += 1
        for a, c in [("v", "M"), ("M", "v"), ("v", "v"), ("M", "M"), ("f", "n"), ("n", "f"), ("v", "f"),
                     ("f", "v"), ("X", "X"), ("vb", "v"), ("(n, n)", "f"), ("3", "b"), ("H", "v"),
                     ("v", "f if b else 1.0"), ("f if b else 1.0", "v"), ("undefined_name", "v"),
                     ("{1: 2}", "v"), ("v", "{1: 2}"), ("L", "Li"), ("z", "f")]:
            lines.append("r%d = %s(%s, %s)" % (i, name, a, c))
            lines.append("print(%s(%s, %s))" % (name, a, c))
            i += 1
        for a in ["vb, v, -v", "v, -1.0, 1.0", "f, 1.0, 2.0", "1, 10, 3", "0.0, 1.0, 5", "f, g, n, 2"]:
            lines.append("r%d = %s(%s)" % (i, name, a))
            i += 1
        for kw in KEYWORDS:
            for val in ["0", "f", "float", "complex", "int", "{1: 2}", "'x'"]:
                lines.append("r%d = %s(v, %s=%s)" % (i, name, kw, val))
                i += 1
        for g in GEN_ARGS:
            lines.append("r%d = %s(%s)" % (i, name, g))
            i += 1
        lines.append("r%d = %s()" % (i, name))
        yield name, lines


class Generated(str):
    """A generated program: its source, and its statements (after PRELUDE), so that a
    statement that stops both translators can be dropped and the rest compared."""

    def __new__(cls, stmts):
        s = super().__new__(cls, PRELUDE + "\n".join(stmts) + "\n")
        s.stmts = stmts
        return s


CHUNK = 20


def chunks(label, lines, filename):
    """A long generated program as programs of CHUNK statements (a statement that stops the
    translator then costs one chunk's worth of per-statement checks, not the program's)."""
    for i in range(0, len(lines), CHUNK):
        yield "%s#%d" % (label, i // CHUNK), Generated(lines[i:i + CHUNK]), filename


def inputs_generated(seed, count):
    names = rule_names()
    for name, lines in gen_rule_programs(names):
        yield from chunks("generated/rule/" + name, lines, "gen_%s.py" % re.sub(r"\W", "_", name))
    rnd = random.Random(seed)
    for j in range(count):
        yield from chunks("generated/random/%04d" % j, gen_program(rnd, names), "gen_%04d.py" % j)
    # methods and attributes on every receiver, and statement forms, spelled out
    lines = []
    i = 0
    for r in RECEIVERS:
        for m in METHODS:
            for args in ["", "0", "v", "float", "np.float64", "int", "v, 1", "axis=0"]:
                lines.append("r%d = %s.%s(%s)" % (i, r, m, args))
                lines.append("%s.%s(%s)" % (r, m, args))
                i += 1
        for a in ATTRS:
            lines.append("r%d = %s.%s" % (i, r, a))
            lines.append("print(%s.%s)" % (r, a))
            i += 1
    for c in CONSTS:
        lines.append("r%d = %s" % (i, c))
        lines.append("print(%s)" % c)
        i += 1
    yield from chunks("generated/methods", lines, "gen_methods.py")
    # unpacking rules: divmod and eigh, with every kind of target
    lines = []
    for call in ["divmod(n, 3)", "divmod(f, 2.0)", "divmod(k, -3)", "divmod(v, 2.0)", "divmod(n)", "divmod(n, 3, 4)",
                 "divmod(a=n, b=3)", "divmod(n, undefined_name)", "divmod(n, {1: 2})", "np.linalg.eigh(H)",
                 "np.linalg.eigh(undefined_name)", "np.linalg.eigvalsh(H)", "np.sort(v)", "np.linalg.qr(H)"]:
        for tgt in ["q, r", "q, r, s", "q, *r", "(q, r), s", "q, M[0, 0]", "q, f.x", "q", "[q, r]"]:
            lines.append("%s = %s" % (tgt, call))
            lines.append("print(q)")
    yield from chunks("generated/unpack", lines, "gen_unpack.py")
    # inside a def that grad differentiates, and the same def not differentiated
    for body in ["np.sum(w)", "np.mean(w)", "w.mean()", "w.sum()", "np.std(w)", "w.var()", "np.var(w)", "w.std()",
                 "sum(w)", "np.dot(w, w)", "np.max(w)", "math.sin(w[0])", "np.sum(w * w) + np.mean(w)",
                 "np.linalg.norm(w)", "np.sum(w, axis=0)", "math.sinh(w[0]) + math.cosh(w[1])"]:
        for diff in ("grad(loss)(v)", "loss(v)"):
            src = "from viv import grad\n" + PRELUDE + "def loss(w):\n    return %s\n\ng = %s\nprint(g)\n" % (body, diff)
            yield "generated/grad/%s/%s" % (body, diff.split("(")[0]), src, "gen_grad.py"
    # math functions of numbers (the guarded forms, a temporary for sinh and cosh)
    lines = []
    for fn in ["sqrt", "exp", "log", "sin", "cos", "tan", "tanh", "sinh", "cosh", "asin", "acos", "atan", "erf",
               "gamma", "floor", "ceil", "fabs", "trunc", "log10", "log2", "log1p", "pow", "fmod", "hypot", "atan2",
               "isclose", "isnan", "isinf", "isfinite"]:
        for a in ["f", "n", "b", "-f", "f if b else 1.0", "v", "z", "v[0] + 1.0", "(f, f)", "f, 2.0", "n, 2", "f, rel_tol=0.1",
                  "f, abs_tol=f if b else 0.0"]:
            lines.append("r = math.%s(%s)" % (fn, a))
            lines.append("print(math.%s(%s))" % (fn, a))
    yield from chunks("generated/math", lines, "gen_math.py")


def without_crashers(base, src, filename):
    """The generated program without the statements that stop the baseline on their own
    (whether a statement stops the translator does not depend on the print mode)."""
    if getattr(src, "pruned", None) is None:
        keep = [st for st in src.stmts if "exception" not in outcome(base, PRELUDE + st + "\n", filename, "exact")]
        src.pruned = Generated(keep)
    return src.pruned, len(src.stmts) - len(src.pruned.stmts)


SOURCES = {"examples": lambda a: inputs_examples(), "posts": lambda a: inputs_posts(), "ta": lambda a: inputs_ta(),
           "tests": lambda a: inputs_tests(), "generated": lambda a: inputs_generated(a.seed, a.random)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="py2viv with rules/*.toml against the translator it replaced")
    ap.add_argument("--base", default=BASE, help="git revision of the baseline py2viv.py (default %s)" % BASE)
    ap.add_argument("--only", default=",".join(SOURCES), help="comma-separated: " + ", ".join(SOURCES))
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--random", type=int, default=400, help="random generated programs (default 400)")
    ap.add_argument("--max-diffs", type=int, default=20)
    args = ap.parse_args(argv)
    base = load_base(args.base)
    print("baseline: tools/py2viv/py2viv.py at %s; Python %s" % (args.base, sys.version.split()[0]))
    total_bad = 0
    shown = 0
    fixed = {}      # fix id -> translations it changed
    examples = {}   # fix id -> one input it changed
    for which in args.only.split(","):
        n = bad = refused = crashed = dropped = changed = 0
        skipped = which == "ta" and sys.version_info < (3, 12)
        for label, src, filename in SOURCES[which](args):
            for pm in MODES:
                a = outcome(base, src, filename, pm)
                b = outcome(py2viv, src, filename, pm)
                if "exception" in a and isinstance(src, Generated):
                    # compare the rest of the program too: the statements that stop the baseline
                    # are set aside (the new translator must not crash on any of them)
                    src2, k = without_crashers(base, src, filename)
                    for st in set(src.stmts) - set(src2.stmts) if pm == MODES[0] else ():
                        o = outcome(py2viv, PRELUDE + st + "\n", filename, pm)
                        if "exception" in o:
                            continue  # both stop (a call Python rejects too)
                        fixed["fixed-crash"] = fixed.get("fixed-crash", 0) + 1
                        examples.setdefault("fixed-crash", "%s: `%s`" % (label, st))
                    dropped += k
                    a = outcome(base, src2, filename, pm)
                    b = outcome(py2viv, src2, filename, pm)
                    src = src2
                n += 1
                if "exception" in a:
                    crashed += 1
                elif a["messages"]:
                    refused += 1
                d = first_diff(a, b)
                if not d:
                    continue
                fixes, where = explain(a, b, str(src))
                if fixes is not None:
                    changed += 1
                    for f in fixes:
                        fixed[f] = fixed.get(f, 0) + 1
                        examples.setdefault(f, label)
                    continue
                bad += 1
                if shown < args.max_diffs:
                    shown += 1
                    print("DIFFERS %s (print %s), explained by no listed fix: %s\n    first difference: %s" % (label, pm, where, d))
        total_bad += bad
        note = " (skipped: TheAlgorithms needs Python 3.12)" if skipped else ""
        drop = ", %d crashing statements set aside" % dropped if dropped else ""
        print("%-10s %6d translations, %6d with refusals, %4d stop in both%s: %s%s" % (
            which, n, refused, crashed, drop,
            ("identical" if not changed else "identical except %d changed by the listed fixes" % changed)
            if not bad else "%d DIFFER unexplained" % bad, note))
    if fixed:
        print("changed translations, per fix (a translation can count under several):")
        for fid, desc, *_ in FIXES:
            if fid in fixed:
                print("  %-17s %6d  e.g. %s" % (fid, fixed[fid], examples[fid][:90]))
    print("RESULT: %s" % ("identical except the listed fixes" if not total_bad else "%d translations differ unexplained" % total_bad))
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
