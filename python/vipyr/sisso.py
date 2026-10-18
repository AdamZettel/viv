# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Runhai Ouyang (SISSO, github.com/rouyang2017/SISSO)
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vipyr.sisso: SISSO symbolic regression, the Viv port of the reference Fortran.

    from vipyr import sisso
    r = sisso.fit(df, target="y", features=["a", "b", "c"],
                  units={"a": "m", "b": "m", "c": "s"},
                  rung=2, dim=2, ops=["+", "-", "*", "/", "^2", "sqrt"])
    r.models[-1].descriptors      # ['((a*b)/c)', '(c)^2']
    r.models[-1].coefficients     # one per descriptor
    r.models[-1].intercept, r.models[-1].rmse
    r.predict(df)                 # the largest dimension's model; dim=1 for another

SISSO (Ouyang et al., Phys. Rev. Materials 2, 083802 (2018)) builds every
formula of the primary features up to a complexity, keeps only those whose
units are consistent, ranks them by correlation with the target (sure
independence screening), and then tries every descriptor of 1..dim of the
ranked formulas by least squares (L0). The computation is
study/ports/sisso/sisso.viv, a port of SISSO 3.5 (Runhai Ouyang,
Apache-2.0; THIRD_PARTY.md), compiled once through vipyr's cache. It follows the
Fortran's order of operations: on the same data it gives the same feature
spaces, subspace and descriptors (study/ports/sisso/compare_fortran.py).
Regression only.
"""
from fractions import Fraction
from pathlib import Path
import re

import numpy as np

SOURCE = Path(__file__).resolve().parents[2] / "study" / "ports" / "sisso" / "sisso.viv"

OPS = {"+": 1, "-": 2, "*": 3, "/": 4, "exp": 5, "exp-": 6, "^-1": 7, "^2": 8, "^3": 9, "sqrt": 10,
       "cbrt": 11, "log": 12, "|-|": 13, "scd": 14, "^6": 15, "sin": 16, "cos": 17}
OP_PRIMARY, OP_NO = 0, 18
_BINARY = {1: "+", 2: "-", 3: "*", 4: "/"}
_PREFIX = {5: "exp(", 6: "exp(-", 14: "scd(", 10: "sqrt(", 11: "cbrt(", 12: "log(", 16: "sin(", 17: "cos("}
_POWER = {7: ")^-1", 8: ")^2", 9: ")^3", 15: ")^6"}
_PI_F = 3.14159265          # the Fortran's PI

_module = None


def _load(fast=False):
    global _module
    if _module is None:
        from . import _core
        _module = _core.load(SOURCE, fast=fast)
    return _module


def parse_ops(ops):
    """Operator names, a list or SISSO's string '(+)(-)(*)(/)', as codes."""
    if isinstance(ops, str):
        ops = re.findall(r"\(([^()]*)\)", ops) or [ops]
    out = []
    for o in ops:
        if o not in OPS:
            raise ValueError("sisso: unknown operator %r; known: %s" % (o, " ".join(OPS)))
        out.append(OPS[o])
    return out


def parse_unit(u):
    """A unit as {base: Fraction}: '', None, 'm', 'm/s', 'kg*m^2/s^2', 'm s^-1', or a dict."""
    if u is None:
        return {}
    if isinstance(u, dict):
        return {k: Fraction(v) for k, v in u.items() if v}
    out = {}
    sign = 1
    for tok in re.findall(r"[*/ ]|[^*/ ]+", str(u)):
        if tok in "* ":
            continue
        if tok == "/":
            sign = -1
            continue
        base, _, p = tok.partition("^")
        if base not in ("1", ""):
            out[base] = out.get(base, 0) + sign * Fraction(p or 1)
        sign = 1
    return {k: v for k, v in out.items() if v}


def rung_complexity(rung):
    """The largest complexity of a rung: SISSO.f90's table read backwards."""
    return 2 ** rung - 1


class Model:
    """One dimension's descriptor: y = intercept + sum(coefficients * descriptors)."""

    def __init__(self, dim, indices, descriptors, coefficients, intercept, rmse, maxae, trees):
        self.dim, self.indices, self.descriptors = dim, indices, descriptors
        self.coefficients, self.intercept, self.rmse, self.maxae = coefficients, intercept, rmse, maxae
        self._trees = trees

    def formula(self, target="y"):
        terms = ["%.10g" % self.intercept] + ["%+.10g*%s" % (c, d) for c, d in zip(self.coefficients, self.descriptors)]
        return "%s = %s" % (target, " ".join(terms))

    def __repr__(self):
        return "<sisso %dD rmse=%.6g: %s>" % (self.dim, self.rmse, self.formula())


class SissoResult:
    """The models of dimensions 1..dim, the feature-space sizes and the subspace."""

    def __init__(self, raw, names, target):
        self.raw, self.names, self.target = raw, names, target
        self.node_op, self.node_a, self.node_b = (np.asarray(raw[k]).astype(np.int64)
                                                  for k in ("NODE_OP", "NODE_A", "NODE_B"))
        us = [tuple(int(v) for v in t) for t in zip(raw["US_OP"], raw["US_A"], raw["US_B"])]
        self.subspace = [self.render(o, a, b) for o, a, b in us]
        self.subspace_scores = list(np.asarray(raw["US_SCORE"]))
        self.sis_sizes = [int(x) for x in raw["SIS_SIZES"]]
        phi = [int(x) for x in raw["PHI_SIZES"]]
        k = len(phi) // len(self.sis_sizes)
        self.phi_sizes = [phi[i * k:(i + 1) * k] for i in range(len(self.sis_sizes))]
        self.models, p = [], 0
        for d in range(1, len(raw["MODEL_C0"]) + 1):
            idx = [int(i) for i in raw["MODEL_IDX"][p:p + d]]
            self.models.append(Model(d, idx, [self.subspace[i] for i in idx],
                                     [float(c) for c in raw["MODEL_COEF"][p:p + d]],
                                     float(raw["MODEL_C0"][d - 1]), float(raw["MODEL_RMSE"][d - 1]),
                                     float(raw["MODEL_MAXAE"][d - 1]), [us[i] for i in idx]))
            p += d

    # names, as the Fortran writes them (FC.f90 combine)
    def render(self, o, a, b):
        o, a, b = int(o), int(a), int(b)
        if o == OP_PRIMARY:
            return self.names[a]
        if o == OP_NO:
            return "(" + self.names[a] + ")"
        sa = self.render(self.node_op[a], self.node_a[a], self.node_b[a])
        if o in _BINARY:
            return "(" + sa + _BINARY[o] + self.render(self.node_op[b], self.node_a[b], self.node_b[b]) + ")"
        if o == 13:
            return "abs(" + sa + "-" + self.render(self.node_op[b], self.node_a[b], self.node_b[b]) + ")"
        if o in _POWER:
            return "(" + sa + _POWER[o]
        return _PREFIX[o] + sa + ")"

    def evaluate(self, o, a, b, cols):
        """The values of the feature (o, a, b) on columns {name: array}."""
        o, a, b = int(o), int(a), int(b)
        if o in (OP_PRIMARY, OP_NO):
            return np.asarray(cols[self.names[a]], dtype=float)
        x = self.evaluate(self.node_op[a], self.node_a[a], self.node_b[a], cols)
        if o in _BINARY or o == 13:
            y = self.evaluate(self.node_op[b], self.node_a[b], self.node_b[b], cols)
            return {1: x + y, 2: x - y, 3: x * y, 4: x / y, 13: np.abs(x - y)}[o]
        with np.errstate(all="ignore"):
            return {5: np.exp(x), 6: np.exp(-x), 7: 1.0 / x, 8: x * x, 9: x * x * x, 15: (x * x * x) ** 2,
                    10: np.sqrt(x), 11: np.cbrt(x), 12: np.log(x), 14: 1.0 / (_PI_F * (1.0 + x * x)),
                    16: np.sin(x), 17: np.cos(x)}[o]

    def descriptors(self, df, dim=None):
        """The descriptor values of a model on new data, one column each."""
        m = self.models[(dim or len(self.models)) - 1]
        cols = _columns(df, self.names)
        return np.column_stack([self.evaluate(o, a, b, cols) for o, a, b in m._trees])

    def predict(self, df, dim=None):
        m = self.models[(dim or len(self.models)) - 1]
        return m.intercept + self.descriptors(df, dim) @ np.asarray(m.coefficients)

    def __repr__(self):
        return "<SissoResult %s>" % "; ".join(repr(m) for m in self.models)


def _columns(df, names):
    return {n: np.asarray(df[n], dtype=float) for n in names}


def fit(df, target="y", features=None, units=None, rung=2, dim=2, ops=("+", "-", "*", "/"), nf_sis=1000,
        fcomplexity=None, fmax_min=1e-3, fmax_max=1e5, max_values=100_000_000, fast=False):
    """SISSO regression of `target` on the `features` columns of `df` (a
    pandas or polars DataFrame, or a dict of arrays).

    units: {column: unit}; a unit is a string ('m/s', 'kg*m^2', 'eV') or
    {base: exponent}; a column not named is dimensionless. Features of
    different units are never added or subtracted.
    rung: the depth of the formula trees; fcomplexity (operators per
    formula, SISSO 3.5's input) overrides it and defaults to 2**rung - 1.
    dim: the largest descriptor dimension. nf_sis: formulas screened into
    the subspace per dimension. fmax_min, fmax_max: formulas whose largest
    magnitude falls outside are kept but not screened, as in SISSO.
    max_values caps the stored space (features x samples) to bound memory.
    """
    p, features = params(df, target=target, features=features, units=units, rung=rung, dim=dim, ops=ops,
                         nf_sis=nf_sis, fcomplexity=fcomplexity, fmax_min=fmax_min, fmax_max=fmax_max,
                         max_values=max_values)
    raw = _load(fast).run(**p)
    return SissoResult({k.upper(): v for k, v in raw.items()}, features, target)


def params(df, target="y", features=None, units=None, rung=2, dim=2, ops=("+", "-", "*", "/"), nf_sis=1000,
           fcomplexity=None, fmax_min=1e-3, fmax_max=1e5, max_values=100_000_000):
    """The program's parameters for fit()'s arguments, and the feature names."""
    if features is None:
        features = [c for c in df.keys() if c != target]
    features = list(features)
    for f in features:
        if any(ch in f for ch in "()+-*/^ "):
            raise ValueError("sisso: feature name %r holds an operator character" % f)
    units = units or {}
    unknown = set(units) - set(features)
    if unknown:
        raise ValueError("sisso: units for columns that are not features: %s" % sorted(unknown))
    parsed = [parse_unit(units.get(f)) for f in features]
    bases = sorted({b for u in parsed for b in u}) or ["1"]
    U = np.array([[float(u.get(b, 0)) for b in bases] for u in parsed])
    cols = _columns(df, features)
    X = np.array([cols[f] for f in features])
    y = np.asarray(df[target], dtype=float)
    codes, starts = [], [0]
    for f in features:
        codes.extend(f.encode())
        starts.append(len(codes))
    params = dict(x_data=X, y_data=y, unit_exps=U, label_codes=codes, label_starts=starts, op_codes=parse_ops(ops),
                  fcomplexity_in=fcomplexity if fcomplexity is not None else rung_complexity(rung),
                  desc_dim_in=dim, nf_sis_in=nf_sis, fmax_min=fmax_min, fmax_max=fmax_max,
                  max_values_in=int(max_values))
    return params, features
