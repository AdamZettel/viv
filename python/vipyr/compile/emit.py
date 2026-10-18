# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""A fitted scikit-learn pipeline as Viv source.

`emit(model, schema, features)` walks the pipeline and writes one Viv
module whose `score(xf, xi, rows)` takes the input columns (floats in the
row-major `xf`, integers and category codes in the flat `xi`) and returns a
`rows x NOUT` matrix: the class probabilities and the predicted class's
index for a classifier, the prediction for a regressor.

Every step is written with the arithmetic scikit-learn does, in its order,
so that the results are its results to the bit where its own code is
deterministic (README of examples/deploy):

- StandardScaler `(x - mean) / scale`; MinMaxScaler `x * scale + min`, then
  the clip; SimpleImputer replaces NaN (or the declared marker) with its
  statistic; OneHotEncoder compares with the fitted categories.
- Trees compare the feature as float32, as scikit-learn's trees do (they
  cast X to float32). `float32(x) <= t` is the same as `x <= T` in float64
  for one T per node, computed here (`le_bound`), so the Viv source compares
  doubles with the bound and no cast runs per node.
- Gradient boosting adds `learning_rate * value` with the fused multiply-add
  when scikit-learn's compiled loop does (detected per model: `fused`).
- Forests add the trees' outputs in tree order from 0.0 and divide by the
  count, as `predict_proba` does with `n_jobs=None`.
- Linear models: the dot product is BLAS's in scikit-learn; here it is the
  plain left-to-right sum, so linear results agree to a stated tolerance,
  not to the bit (compile.verify measures it).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np


#: above this many nodes in all, "auto" writes the trees as tables: the
#: branch code grows with the node count (about 100 bytes of C per node; a
#: 500-tree, depth-8 model of 115,484 nodes is 11 MB of C, built in 8 s)
#: and so does the C compiler's time
BRANCH_NODES = 250_000


class CompileError(ValueError):
    """The model holds something this compiler does not translate."""


# ----------------------------------------------------------------- numbers

def num(x) -> str:
    """A float64 as a Viv literal that reads back to the same bits."""
    x = float(x)
    if math.isnan(x):
        raise CompileError("a NaN constant in the model (a statistic of an all-missing column?)")
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    r = repr(x)
    if "e" not in r and "." not in r and "inf" not in r:
        r += ".0"
    if "e" in r and "." not in r.split("e")[0]:
        a, b = r.split("e")
        r = a + ".0e" + b
    return r


def le_bound(threshold: float) -> float:
    """The float64 T with `float32(x) <= threshold` exactly when `x <= T`,
    for every float64 x that is not NaN.

    float32 rounding is monotone, so the floats x reaches with
    `float32(x) <= threshold` are those at most F, the largest float32 not
    above the threshold, and x rounds to at most F below the midpoint m of F
    and the next float32; at m itself, ties go to the even significand. So
    T is m when F is even, and the double just below m when F is odd."""
    t = float(threshold)
    f = np.float32(t)
    if float(f) > t:
        f = np.nextafter(f, np.float32(-np.inf), dtype=np.float32)
    if np.isinf(f):
        # below -FLT_MAX: x rounds to -inf at or below the midpoint of -inf
        # (read as -2^128) and -FLT_MAX, the tie going to -inf (even)
        return -(2.0 ** 128 - 2.0 ** 103)
    up = np.nextafter(f, np.float32(np.inf), dtype=np.float32)
    up_d = 2.0 ** 128 if np.isinf(up) else float(up)
    mid = (float(f) + up_d) / 2.0          # exact: 25 significant bits
    even = (int(np.array(f, dtype=np.float32).view(np.uint32)) & 1) == 0
    return mid if even else float(np.nextafter(mid, -np.inf))


# ------------------------------------------------------------------ schema

@dataclass
class Column:
    """One input column: `float` (a double), `int` (an int64) or `category`
    (a string, read as its position in `categories`; -1 when unknown)."""
    name: str
    kind: str
    categories: list = None
    slot: int = 0                         # its index in xf (float) or xi


@dataclass
class FeatureDef:
    """A def of the features module: parameters and results by column name."""
    name: str
    params: list                          # [(name, "float"|"int")]
    outputs: list                         # [(name, "float"|"int")]
    record: str = None                    # None: one array result, named like the def


def read_features(text: str):
    """The feature defs of a Viv module: each public top-level def whose
    parameters are all `Arr` or `[Int]` and whose result is an array or a
    record of arrays. Read from the source text: the signatures only."""
    records = {}
    for m in re.finditer(r"^record\s+(\w+)\s*\{(.*?)^\}", text, re.M | re.S):
        fields = []
        for fm in re.finditer(r"^\s*(\w+)\s*:\s*(\[Int\]|Arr)\s*$", m.group(2), re.M):
            fields.append((fm.group(1), "int" if fm.group(2) == "[Int]" else "float"))
        records[m.group(1)] = fields
    defs = []
    for m in re.finditer(r"^def\s+(\w+)\s*\(([^)]*)\)\s*->\s*(\[Int\]|Arr|\w+)\s*\{", text, re.M):
        name, params, ret = m.group(1), m.group(2), m.group(3)
        if name.startswith("_"):
            continue
        ps = []
        ok = True
        for p in [p.strip() for p in params.split(",") if p.strip()]:
            pm = re.fullmatch(r"(\w+)\s*:\s*(\[Int\]|Arr)", p)
            if not pm:
                ok = False
                break
            ps.append((pm.group(1), "int" if pm.group(2) == "[Int]" else "float"))
        if not ok or not ps:
            continue
        if ret in ("[Int]", "Arr"):
            defs.append(FeatureDef(name, ps, [(name, "int" if ret == "[Int]" else "float")]))
        elif ret in records and records[ret]:
            defs.append(FeatureDef(name, ps, records[ret], record=ret))
    return defs


# ------------------------------------------------------------- the writer

class Writer:
    def __init__(self):
        self.consts = []        # module-level lines
        self.pre = []           # lines of score() before the row loop
        self.row = []           # lines inside the transform loop
        self.est = []           # lines inside the estimator loop
        self.defs = []          # helper defs
        self.notes = []         # what README / the report says about the model
        self.n = 0
        self.tree_code = "auto"  # trees as "branches", "tables", or "auto" by size
        self.tree_mode = None   # what was chosen
        self.z_used = set()     # the features the tree defs read, as z0, z1, ...
        self.uses_z = False     # whether the estimator reads the matrix Z
        self.updates = []       # an ensemble's per-tree statements, in tree order
        self.accs = []          # the per-row sums the updates add to
        self.split = None       # where in self.est the updates go
        self.n_nodes = 0        # the trees' nodes in all

    def tmp(self, base="v"):
        self.n += 1
        return "%s%d" % (base, self.n)


def _ident(s: str) -> str:
    s = re.sub(r"\W", "_", str(s))
    return ("c_" + s) if (not s or s[0].isdigit()) else s


def _arr(vals, per_line=6):
    vals = list(vals)
    if not vals:
        return "[]"
    parts = [", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    if len(parts) == 1:
        return "[" + parts[0] + "]"
    return "[\n    " + ",\n    ".join(parts) + ",\n]"


# A value in the row loop: ("num", var) a Num, ("int", var) an Int,
# ("code", var, column) a category's position in the column's categories.

def _as_num(w, val):
    if val[0] == "num":
        return val[1]
    if val[0] == "int":
        return "float(%s)" % val[1]
    raise CompileError("column %r holds strings; only a OneHotEncoder may read it" % val[2].name)


def _cls(o):
    return type(o).__name__


# ----------------------------------------------------------- transformers

def _transform(w, est, vals, names):
    """`est` applied to the row's values `vals` (one per column); returns the
    output values and their names."""
    from sklearn.pipeline import Pipeline
    if est is None or est == "passthrough":
        return vals, names
    if isinstance(est, Pipeline):
        for _, step in est.steps:
            vals, names = _transform(w, step, vals, names)
        return vals, names
    k = _cls(est)
    if k == "StandardScaler":
        w.row.append("#: StandardScaler on %s: (x - mean) / scale" % ", ".join(names))
        out = []
        for j, v in enumerate(vals):
            e = _as_num(w, v)
            if est.with_mean:
                e = "(%s - %s)" % (e, num(est.mean_[j]))
            if est.with_std:
                e = "%s / %s" % (e, num(est.scale_[j]))
            t = w.tmp()
            w.row.append("%s = %s" % (t, e))
            out.append(("num", t))
        return out, names
    if k == "MinMaxScaler":
        lo, hi = est.feature_range
        w.row.append("#: MinMaxScaler on %s: x * scale + min%s" % (", ".join(names), ", clipped" if est.clip else ""))
        out = []
        for j, v in enumerate(vals):
            t = w.tmp()
            w.row.append("%s = %s * %s + %s" % (t, _as_num(w, v), num(est.scale_[j]), num(est.min_[j])))
            if est.clip:
                w.row.append("%s = min(max(%s, %s), %s)" % (t, t, num(lo), num(hi)))
            out.append(("num", t))
        return out, names
    if k == "SimpleImputer":
        if getattr(est, "add_indicator", False):
            raise CompileError("SimpleImputer(add_indicator=True) is not supported")
        mv = est.missing_values
        stats = np.asarray(est.statistics_)
        keep = getattr(est, "keep_empty_features", False)
        w.row.append("#: SimpleImputer(%s) on %s: a missing value becomes the statistic" % (est.strategy, ", ".join(names)))
        out, onames = [], []
        for j, v in enumerate(vals):
            s = stats[j]
            try:
                s = float(s)
            except (TypeError, ValueError):
                raise CompileError("SimpleImputer with a non-numeric fill (%r) is not supported" % (s,))
            if math.isnan(s) and not keep:
                continue      # scikit-learn drops a column with no statistic
            x = _as_num(w, v)
            t = w.tmp()
            miss = "%s != %s" % (x, x) if (isinstance(mv, float) and math.isnan(mv)) or mv is None or mv is np.nan else "%s == %s" % (x, num(mv))
            w.row.append("%s = %s if %s else %s" % (t, num(0.0 if math.isnan(s) else s), miss, x))
            out.append(("num", t))
            onames.append(names[j])
        return out, onames
    if k == "OneHotEncoder":
        return _onehot(w, est, vals, names)
    raise CompileError("transformer %s is not supported (supported: StandardScaler, MinMaxScaler, SimpleImputer, "
                       "OneHotEncoder, Pipeline, ColumnTransformer)" % k)


def _onehot(w, est, vals, names):
    if getattr(est, "sparse_output", getattr(est, "sparse", False)) not in (False,):
        pass   # the ColumnTransformer's sparse_output_ is what reaches the estimator; checked there
    if est.dtype not in (np.float64, float, "float64"):
        raise CompileError("OneHotEncoder(dtype=%r): only float64 output is supported" % (est.dtype,))
    infreq = getattr(est, "infrequent_categories_", None)
    if infreq is not None and any(c is not None for c in infreq):
        raise CompileError("OneHotEncoder with infrequent categories is not supported")
    drop_idx = getattr(est, "drop_idx_", None)
    unknown = est.handle_unknown
    out, onames = [], []
    for j, v in enumerate(vals):
        cats = list(est.categories_[j])
        drop = None if drop_idx is None or drop_idx[j] is None else int(drop_idx[j])
        w.row.append("#: OneHotEncoder on %s: %d categories%s, unknown -> %s" % (
            names[j], len(cats), "" if drop is None else " (drops %r)" % (cats[drop],),
            "an error" if unknown == "error" else "all zeros"))
        if v[0] == "code":
            col = v[2]
            if [str(c) for c in col.categories] != [str(c) for c in cats]:
                raise CompileError("column %r: its categories do not match the encoder's" % col.name)
            code = v[1]
        else:
            # a numeric column: its position among the fitted categories
            x = _as_num(w, v)
            code = w.tmp("code")
            w.row.append("%s: Int = -1" % code)
            for c, cv in enumerate(cats):
                cvf = float(cv)
                test = "%s != %s" % (x, x) if math.isnan(cvf) else "%s == %s" % (x, num(cvf))
                w.row.append("if %s {\n    %s = %d\n}" % (test, code, c))
        if unknown == "error":
            w.row.append('if %s < 0 {\n    stop(f"row {i}: column %s holds a category the encoder did not see in training '
                         '(handle_unknown=\'error\')")\n}' % (code, names[j]))
        for c, cv in enumerate(cats):
            if c == drop:
                continue
            t = w.tmp()
            w.row.append("%s = 1.0 if %s == %d else 0.0" % (t, code, c))
            out.append(("num", t))
            onames.append("%s=%s" % (names[j], cv))
    return out, onames


def _column_transformer(w, ct, avail):
    if getattr(ct, "sparse_output_", False):
        raise CompileError("the ColumnTransformer's output is sparse: set sparse_threshold=0 or "
                           "OneHotEncoder(sparse_output=False)")
    fnames = list(ct.feature_names_in_)
    idx = ct._transformer_to_input_indices
    out, onames = [], []
    for name, trans, _cols in ct.transformers_:
        cols = idx.get(name, [])
        if trans == "drop" or len(cols) == 0:
            continue
        cn = [fnames[c] for c in cols]
        vals = [avail[c_] for c_ in cn]
        w.row.append("#: %s" % name)
        o, n = _transform(w, None if trans == "passthrough" else trans, vals, cn)
        out += o
        onames += n
    return out, onames


# ------------------------------------------------------------- estimators

def _flatten_trees(trees, value_fn, sklearn_f32=True):
    """The nodes of `trees` (sklearn Tree objects) as flat arrays: node j of
    tree t is index roots[t] + j."""
    feat, le, left, right, nanl, vals, roots = [], [], [], [], [], [], []
    base = 0
    for tr in trees:
        roots.append(base)
        n = tr.node_count
        mgl = getattr(tr, "missing_go_to_left", None)
        v = value_fn(tr)
        for j in range(n):
            leaf = tr.children_left[j] == -1
            feat.append(-1 if leaf else int(tr.feature[j]))
            le.append(0.0 if leaf else (le_bound(tr.threshold[j]) if sklearn_f32 else float(tr.threshold[j])))
            left.append(-1 if leaf else base + int(tr.children_left[j]))
            right.append(-1 if leaf else base + int(tr.children_right[j]))
            nanl.append(0 if (leaf or mgl is None) else int(mgl[j]))
            vals.append(v[j])
        base += n
    return dict(feat=feat, le=le, left=left, right=right, nanl=nanl, vals=vals, roots=roots)


def _emit_tree_tables(w, prefix, t, k, nan_aware):
    w.consts.append("#: %s: %d trees, %d nodes. Node j: feature %s_FEAT[j] (-1 at a leaf); go left when the "
                    "feature is at most %s_LE[j]" % (prefix, len(t["roots"]), len(t["feat"]), prefix, prefix))
    w.consts.append("%s_ROOT: [Int] = %s" % (prefix, _arr([str(x) for x in t["roots"]], 12)))
    w.consts.append("%s_FEAT: [Int] = %s" % (prefix, _arr([str(x) for x in t["feat"]], 16)))
    w.consts.append("%s_LE: Arr = %s" % (prefix, _arr([num(x) for x in t["le"]], 4)))
    w.consts.append("%s_LEFT: [Int] = %s" % (prefix, _arr([str(x) for x in t["left"]], 12)))
    w.consts.append("%s_RIGHT: [Int] = %s" % (prefix, _arr([str(x) for x in t["right"]], 12)))
    if nan_aware:
        w.consts.append("%s_NANLEFT: [Int] = %s" % (prefix, _arr([str(x) for x in t["nanl"]], 24)))
    flat = [num(x) for row in t["vals"] for x in (row if isinstance(row, (list, tuple, np.ndarray)) else [row])]
    w.consts.append("#: the value of node j, %d per node, at %s_VALUE[j * %d + k]" % (k, prefix, k))
    w.consts.append("%s_VALUE: Arr = %s" % (prefix, _arr(flat, 4)))
    nan_lines = ("        if x != x {\n            if %s_NANLEFT[node] == 1 {\n                node = %s_LEFT[node]\n            }\n"
                 "            else {\n                node = %s_RIGHT[node]\n            }\n        }\n"
                 "        else if x <= %s_LE[node] {" % (prefix, prefix, prefix, prefix)) if nan_aware else \
        "        if x <= %s_LE[node] {" % prefix
    w.defs.append(
        "#| The leaf of %s's tree whose first node is `node0`, for row i of Z.\n"
        "def _%s_leaf(Z: Arr, i: Int, node0: Int) -> Int {\n"
        "    node: Int = node0\n"
        "    while %s_FEAT[node] >= 0 {\n"
        "        x = Z[i, %s_FEAT[node]]\n"
        "%s\n"
        "            node = %s_LEFT[node]\n"
        "        }\n"
        "        else {\n"
        "            node = %s_RIGHT[node]\n"
        "        }\n"
        "    }\n"
        "    return node\n"
        "}\n" % (prefix, prefix.lower(), prefix, prefix, nan_lines, prefix, prefix))


def _emit_tree_branches(w, t, nan_aware, want_leaf):
    """Each tree as a def of nested ifs whose thresholds are constants: the
    def takes the features its tree reads (z3, z7, ...) and returns the
    leaf's value, or with `want_leaf` the leaf's node index into T_VALUE.
    The comparisons are those of the tables (`x <= T`; a NaN goes left
    where missing_go_to_left says so, otherwise right), so the leaf reached
    is the same. Returns one call per tree, in tree order."""
    feat, le, left, right, nanl, vals = t["feat"], t["le"], t["left"], t["right"], t["nanl"], t["vals"]
    calls = []
    for k, root in enumerate(t["roots"]):
        used = set()

        def walk(node, ind):
            pad = " " * ind
            if feat[node] < 0:
                return [pad + "return %s" % (str(node) if want_leaf else num(vals[node]))]
            f = feat[node]
            used.add(f)
            cond = "z%d <= %s" % (f, num(le[node]))
            if nan_aware and nanl[node]:
                cond += " or z%d != z%d" % (f, f)
            return ([pad + "if %s {" % cond] + walk(left[node], ind + 4) + [pad + "}", pad + "else {"]
                    + walk(right[node], ind + 4) + [pad + "}"])
        body = walk(root, 4)
        args = sorted(used)
        name = "_tree%d" % k
        w.defs.append("def %s(%s) -> %s {\n%s\n}\n" % (
            name, ", ".join("z%d: Num" % j for j in args), "Int" if want_leaf else "Num", "\n".join(body)))
        w.z_used.update(args)
        calls.append("%s(%s)" % (name, ", ".join("z%d" % j for j in args)))
    return calls


def _trees(w, prefix, t, k, nan_aware, want_leaf):
    """The trees of `t` as code (`branches`) or as node tables walked by
    _t_leaf (`tables`), as w.tree_code says; returns one expression per
    tree: its leaf's value, or with `want_leaf` (or k > 1) the leaf's node
    index into T_VALUE."""
    want_leaf = want_leaf or k > 1
    mode = w.tree_code
    if mode == "auto":
        mode = "branches" if len(t["feat"]) <= BRANCH_NODES else "tables"
    w.tree_mode = mode
    w.n_nodes = len(t["feat"])
    if mode == "branches":
        if want_leaf:
            flat = [num(x) for row in t["vals"] for x in (row if isinstance(row, (list, tuple, np.ndarray)) else [row])]
            w.consts.append("#: the value of node j, %d per node, at %s_VALUE[j * %d + k]" % (k, prefix, k))
            w.consts.append("%s_VALUE: Arr = %s" % (prefix, _arr(flat, 4)))
        return _emit_tree_branches(w, t, nan_aware, want_leaf)
    _emit_tree_tables(w, prefix, t, k, nan_aware)
    w.uses_z = True
    calls = ["_%s_leaf(Z, i, %s_ROOT[%d])" % (prefix.lower(), prefix, j) for j in range(len(t["roots"]))]
    if want_leaf:
        return calls
    return ["%s_VALUE[%s]" % (prefix, c) for c in calls]


def _argmax(w, vars_, out):
    """The first index of the largest of vars_ (NumPy's argmax; NaN wins)."""
    w.est.append("%s: Int = 0" % out)
    best = w.tmp("best")
    w.est.append("%s = %s" % (best, vars_[0]))
    for k in range(1, len(vars_)):
        w.est.append("if %s > %s or (%s != %s and %s == %s) {\n    %s = %d\n    %s = %s\n}" % (
            vars_[k], best, vars_[k], vars_[k], best, best, out, k, best, vars_[k]))
    # NaN: NumPy's argmax returns the first NaN
    return out


def _softmax(w, raw):
    """sklearn.utils.extmath.softmax on one row: subtract the max, exp,
    divide by the sum (NumPy's pairwise order, left to right below 8)."""
    k = len(raw)
    m = w.tmp("mx")
    w.est.append("%s = %s" % (m, raw[0]))
    for r in raw[1:]:
        w.est.append("%s = %s if %s > %s else %s" % (m, r, r, m, m))   # np.max: NaN aside
    es = []
    for r in raw:
        e = w.tmp("e")
        w.est.append("%s = exp(%s - %s)" % (e, r, m))
        es.append(e)
    s = w.tmp("sm")
    # sum() adds in NumPy's order (numpy-agreement), which X.sum(axis=1) uses per row
    w.est.append("%s = sum([%s])" % (s, ", ".join(es)))
    ps = []
    for e in es:
        p = w.tmp("p")
        w.est.append("%s = %s / %s" % (p, e, s))
        ps.append(p)
    return ps


def _expit_pair(w, raw):
    """HalfBinomialLoss.predict_proba: p1 = expit(raw) = 1 / (1 + exp(-raw)), p0 = 1 - p1."""
    p1, p0 = w.tmp("p"), w.tmp("p")
    w.est.append("%s = 1.0 / (1.0 + exp(-%s))" % (p1, raw))
    w.est.append("%s = 1.0 - %s" % (p0, p1))
    return [p0, p1]


def _estimator(w, est, nz, fused):
    """The final estimator's lines; returns (n_out, kind, classes, extra)."""
    k = _cls(est)
    is_clf = hasattr(est, "classes_")
    if hasattr(est, "n_outputs_") and est.n_outputs_ != 1:
        raise CompileError("%s with %d outputs: only one output is supported" % (k, est.n_outputs_))

    if k in ("DecisionTreeClassifier", "DecisionTreeRegressor", "RandomForestClassifier", "RandomForestRegressor",
             "ExtraTreesClassifier", "ExtraTreesRegressor"):
        trees = [est] if k.startswith("DecisionTree") else list(est.estimators_)
        tr = [t.tree_ for t in trees]
        if is_clf:
            K = len(est.classes_)

            # scikit-learn before 1.4 divides the leaf's counts by their sum in
            # predict_proba; from 1.4 the tree stores the fractions themselves
            import inspect
            from sklearn.tree import DecisionTreeClassifier as _DTC
            normalize = "normalizer" in inspect.getsource(_DTC.predict_proba)

            def vf(t):
                v = t.value[:, 0, :K]
                if not normalize:
                    return list(v)
                norm = v.sum(axis=1)[:, np.newaxis]
                norm[norm == 0.0] = 1.0
                return list(v / norm)
        else:
            K = 1

            def vf(t):
                return list(t.value[:, 0, 0])
        tab = _flatten_trees(tr, vf)
        nan_aware = any(x for x in tab["nanl"])
        calls = _trees(w, "T", tab, K, nan_aware, is_clf and k.startswith("DecisionTree"))
        w.notes.append("%s: %d trees, %d nodes, as %s; float32 comparisons as bounds in float64" % (
            k, len(tr), len(tab["feat"]), w.tree_mode))
        nt = len(tr)
        if k.startswith("DecisionTree"):
            accs = []
            if is_clf or K > 1:
                leaf = w.tmp("leaf")
                w.est.append("%s: Int = %s" % (leaf, calls[0]))
                for c in range(K):
                    a = w.tmp("acc")
                    w.est.append("%s = T_VALUE[%s * %d + %d]" % (a, leaf, K, c))
                    accs.append(a)
            else:
                a = w.tmp("acc")
                w.est.append("%s = %s" % (a, calls[0]))
                accs.append(a)
            if is_clf:
                # DecisionTreeClassifier.predict: argmax of the leaf's counts (precomputed per leaf)
                w.consts.append("#: the class DecisionTreeClassifier.predict gives at each node: argmax of its counts")
                cls = [int(np.argmax(est.tree_.value[j, 0, :K])) for j in range(est.tree_.node_count)]
                w.consts.append("T_CLASS: [Int] = %s" % _arr([str(c) for c in cls], 24))
                idx = w.tmp("cls")
                w.est.append("%s: Int = T_CLASS[%s]" % (idx, leaf))
                return K, accs, idx
            return K, accs, None
        accs = []
        for c in range(K):
            a = w.tmp("acc")
            w.est.append("%s = 0.0" % a)
            accs.append(a)
        w.est.append("#: the trees in order, as predict with n_jobs=None adds them, then / %d" % nt)
        w.accs, w.split = accs, len(w.est)
        if K == 1:
            for c_ in calls:
                w.updates.append(["%s += %s" % (accs[0], c_)])
        else:
            leaf = w.tmp("leaf")
            for c_ in calls:
                w.updates.append(["%s = %s" % (leaf, c_)] +
                                 ["%s += T_VALUE[%s * %d + %d]" % (a, leaf, K, c) for c, a in enumerate(accs)])
        outs = []
        for a in accs:
            p = w.tmp("p")
            w.est.append("%s = %s / %s" % (p, a, num(float(nt))))
            outs.append(p)
        if is_clf:
            idx = _argmax(w, outs, w.tmp("cls"))
            return K, outs, idx
        return 1, outs, None

    if k in ("GradientBoostingClassifier", "GradientBoostingRegressor"):
        init = est.init_
        if not (init == "zero" or _cls(init) in ("DummyClassifier", "DummyRegressor")):
            raise CompileError("GradientBoosting with init=%s is not supported" % _cls(init))
        loss = getattr(est, "loss", None)
        n_trees_per = est.estimators_.shape[1]
        if not is_clf and loss not in ("squared_error", "absolute_error", "huber", "quantile"):
            raise CompileError("GradientBoostingRegressor(loss=%r) is not supported" % loss)
        if is_clf and loss not in ("log_loss",):
            raise CompileError("GradientBoostingClassifier(loss=%r) is not supported" % loss)
        trees = [est.estimators_[s, kk].tree_ for s in range(est.estimators_.shape[0]) for kk in range(n_trees_per)]
        tab = _flatten_trees(trees, lambda t: list(t.value[:, 0, 0]))
        calls = _trees(w, "T", tab, 1, False, False)
        # the same raw value for every row: computed by scikit-learn itself
        init_raw = est._raw_predict_init(np.zeros((1, nz), dtype=np.float32))[0]
        lr = float(est.learning_rate)
        w.consts.append("LEARNING_RATE = %s" % num(lr))
        w.consts.append("#: the initial raw prediction (the prior's, from scikit-learn's _raw_predict_init)")
        w.consts.append("INIT_RAW: Arr = %s" % _arr([num(x) for x in init_raw]))
        raw = []
        for kk in range(n_trees_per):
            r = w.tmp("raw")
            w.est.append("%s = INIT_RAW[%d]" % (r, kk))
            raw.append(r)
        w.est.append("#: each stage adds learning_rate * the leaf's value%s" % (
            ", fused into one rounding as scikit-learn's compiled loop does on this machine" if fused else ""))
        w.accs, w.split = raw, len(w.est)
        for s_ in range(est.estimators_.shape[0]):
            for kk, r in enumerate(raw):
                c_ = calls[s_ * n_trees_per + kk]
                w.updates.append(["%s = fma(LEARNING_RATE, %s, %s)" % (r, c_, r) if fused else
                                  "%s = %s + LEARNING_RATE * %s" % (r, r, c_)])
        w.notes.append("%s: %d stages x %d trees, %d nodes, as %s; update %s" % (
            k, est.estimators_.shape[0], n_trees_per, len(tab["feat"]), w.tree_mode,
            "fma(lr, v, raw)" if fused else "raw + lr * v"))
        if not is_clf:
            return 1, raw, None
        if n_trees_per == 1:
            ps = _expit_pair(w, raw[0])
            idx = w.tmp("cls")
            w.est.append("%s: Int = 0\nif %s >= 0.0 {\n    %s = 1\n}" % (idx, raw[0], idx))
            return 2, ps, idx
        ps = _softmax(w, raw)
        idx = _argmax(w, raw, w.tmp("cls"))
        return len(raw), ps, idx

    if k in ("HistGradientBoostingClassifier", "HistGradientBoostingRegressor"):
        if getattr(est, "is_categorical_", None) is not None and np.any(est.is_categorical_):
            raise CompileError("HistGradientBoosting with categorical features is not supported")
        loss = type(est._loss).__name__
        if not is_clf and loss != "HalfSquaredError":
            raise CompileError("HistGradientBoostingRegressor with loss %s is not supported" % loss)
        preds = est._predictors
        K = len(preds[0])
        flat = []
        for it in preds:
            for p in it:
                flat.append(p.nodes)
        feat, le, left, right, nanl, vals, roots = [], [], [], [], [], [], []
        base = 0
        for nodes in flat:
            roots.append(base)
            for j in range(len(nodes)):
                nd = nodes[j]
                leaf = bool(nd["is_leaf"])
                feat.append(-1 if leaf else int(nd["feature_idx"]))
                le.append(0.0 if leaf else float(nd["num_threshold"]))
                left.append(-1 if leaf else base + int(nd["left"]))
                right.append(-1 if leaf else base + int(nd["right"]))
                nanl.append(0 if leaf else int(nd["missing_go_to_left"]))
                vals.append(float(nd["value"]))
            base += len(nodes)
        tab = dict(feat=feat, le=le, left=left, right=right, nanl=nanl, vals=vals, roots=roots)
        calls = _trees(w, "T", tab, 1, True, False)
        base_pred = np.asarray(est._baseline_prediction).reshape(-1)
        w.consts.append("#: the baseline raw prediction")
        w.consts.append("BASELINE: Arr = %s" % _arr([num(x) for x in base_pred]))
        raw = []
        for kk in range(K):
            r = w.tmp("raw")
            w.est.append("%s = BASELINE[%d]" % (r, kk))
            raw.append(r)
        w.est.append("#: each iteration adds its trees' leaf values (comparisons in float64: the thresholds are bin edges)")
        w.accs, w.split = raw, len(w.est)
        for s_ in range(len(preds)):
            for kk, r in enumerate(raw):
                w.updates.append(["%s += %s" % (r, calls[s_ * K + kk])])
        w.notes.append("%s: %d iterations x %d trees, %d nodes, as %s" % (k, len(preds), K, len(feat), w.tree_mode))
        if not is_clf:
            return 1, raw, None
        if K == 1:
            ps = _expit_pair(w, raw[0])
        else:
            ps = _softmax(w, raw)
        idx = _argmax(w, ps, w.tmp("cls"))
        return len(ps), ps, idx

    if k in ("LogisticRegression", "LinearRegression", "Ridge", "Lasso", "ElasticNet"):
        coef = np.atleast_2d(np.asarray(est.coef_, dtype=np.float64))
        icpt = np.atleast_1d(np.asarray(est.intercept_, dtype=np.float64))
        if icpt.size == 1 and coef.shape[0] > 1:
            icpt = np.repeat(icpt, coef.shape[0])
        w.consts.append("#: %s: coefficients, one row per output, and intercepts" % k)
        w.consts.append("COEF: Arr = %s" % _arr([("[" + ", ".join(num(x) for x in r) + "]") for r in coef], 1))
        w.consts.append("INTERCEPT: Arr = %s" % _arr([num(x) for x in icpt]))
        w.uses_z = True
        dec = []
        for r in range(coef.shape[0]):
            d = w.tmp("dec")
            w.est.append("%s = 0.0" % d)
            w.est.append("for j to %d {\n    %s += Z[i, j] * COEF[%d, j]\n}" % (coef.shape[1], d, r))
            w.est.append("%s = %s + INTERCEPT[%d]" % (d, d, r))
            dec.append(d)
        w.notes.append("%s: %d x %d coefficients; the dot product sums left to right (scikit-learn's is BLAS's)" % (
            k, coef.shape[0], coef.shape[1]))
        if not is_clf:
            return 1, dec, None
        if k != "LogisticRegression":
            raise CompileError("%s is not supported" % k)
        multi = getattr(est, "multi_class", "auto")
        n_cls = len(est.classes_)
        if n_cls == 2:
            ps = _expit_pair(w, dec[0])
            idx = w.tmp("cls")
            w.est.append("%s: Int = 0\nif %s > 0.0 {\n    %s = 1\n}" % (idx, dec[0], idx))
            return 2, ps, idx
        ovr = multi == "ovr" or (multi == "auto" and est.solver == "liblinear")
        if ovr:
            raise CompileError("LogisticRegression one-vs-rest with more than two classes is not supported")
        ps = _softmax(w, dec)
        idx = _argmax(w, dec, w.tmp("cls"))
        return n_cls, ps, idx

    raise CompileError("estimator %s is not supported" % k)


# ------------------------------------------------------------------- emit

#: rows per block when an ensemble's trees are written as code: each tree
#: runs over the block's rows before the next tree starts, so one tree's
#: code stays in the instruction cache (a 500-tree, depth-8 model is 11 MB
#: of C, far beyond it) while the block's features stay in the data cache
ROW_BLOCK = 128
#: from this many nodes in all, an ensemble written as code runs in blocks
#: of rows; below it, row by row (the code fits in the cache, and the
#: block's arrays cost as much as they save: equal times measured up to
#: 31,000 nodes, half the time with blocks at 115,000)
BLOCK_NODES = 50_000
#: trees per def in that layout: one def holding every tree's loop took the C
#: compiler 54 s on a 500-tree, depth-8 model
TREES_PER_DEF = 16


def _blocked(w, block):
    """The score loop for an ensemble written as code: rows in blocks of
    ROW_BLOCK; per block, the transformers row by row, then each tree over
    the block's rows, then each row's outputs. Every row's sums take the
    trees' values in tree order, as row by row, so the results are the
    same bits."""
    zs = sorted(w.z_used)
    zsub = re.compile(r"\b(z\d+)\b")

    def per_row(line):
        line = zsub.sub(r"\1b[r]", line)
        for a in w.accs:
            line = re.sub(r"\b%s\b" % a, "%s_b[r]" % a, line)
        return line
    first = ["i = b0 + r"] + w.row + ["#: the estimator"] + w.est[:w.split]
    first += ["z%db[r] = z%d" % (j, j) for j in zs] + ["%s_b[r] = %s" % (a, a) for a in w.accs]
    out = ["#: rows in blocks of ROW_BLOCK; in a block, each tree runs over all its rows in turn",
           "ROW_BLOCK: Int = %d" % ROW_BLOCK]
    out += ["z%db = zeros(ROW_BLOCK)" % j for j in zs] + ["%s_b = zeros(ROW_BLOCK)" % a for a in w.accs]
    inner = ["nb: Int = min(ROW_BLOCK, rows - b0)",
             "#: the transformers, row by row",
             "for r to nb {\n%s\n}" % block(first, 4),
             "#: the trees, one at a time over the block, TREES_PER_DEF to a def"]
    arrays = ["z%db" % j for j in zs] + ["%s_b" % a for a in w.accs]
    for g in range(0, len(w.updates), TREES_PER_DEF):
        name = "_trees%d" % (g // TREES_PER_DEF)
        loops = ["for r to nb {\n%s\n}" % block([per_row(l) for l in u], 4)
                 for u in w.updates[g:g + TREES_PER_DEF]]
        w.defs.append("#| Trees %d to %d over the block's first nb rows: adds to the sums in place.\n"
                      "def %s(%s, nb: Int) {\n%s\n}\n" % (
                          g, min(g + TREES_PER_DEF, len(w.updates)) - 1, name,
                          ", ".join("%s: Arr" % v for v in arrays), block(loops, 4)))
        inner.append("%s(%s, nb)" % (name, ", ".join(arrays)))
    last = ["i = b0 + r"] + ["%s = %s_b[r]" % (a, a) for a in w.accs] + w.est[w.split:]
    inner += ["#: each row's outputs", "for r to nb {\n%s\n}" % block(last, 4)]
    out.append("for b0 from 0 to rows by ROW_BLOCK {\n%s\n}" % block(inner, 4))
    return out


@dataclass
class Emitted:
    source: str
    schema: list
    n_out: int
    classes: list
    is_classifier: bool
    notes: list = field(default_factory=list)
    features_module: str = None


def emit(model, schema, features=None, feature_module_name=None, nan_policy=None, fused=False, title="model",
         tree_code="auto"):
    """Viv source for `model` (a fitted Pipeline or estimator).

    schema: the input Columns, in the order of the CSV and the C entry.
    features: FeatureDefs to run first (from `feature_module_name`).
    nan_policy: {column: (refuse_nan, refuse_inf)} for float columns.
    fused: the gradient-boosting update is a fused multiply-add.
    tree_code: trees as "branches" (a def of nested ifs per tree), "tables"
    (node arrays walked by one loop), or "auto" (branches up to
    BRANCH_NODES nodes in all)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    w = Writer()
    w.tree_code = tree_code
    nan_policy = nan_policy or {}
    floats = [c for c in schema if c.kind == "float"]
    ints = [c for c in schema if c.kind != "float"]
    for j, c in enumerate(floats):
        c.slot = j
    for j, c in enumerate(ints):
        c.slot = j
    NF, NI = len(floats), len(ints)

    steps = list(model.steps) if isinstance(model, Pipeline) else [("model", model)]
    final = steps[-1][1]
    pre = [s for _, s in steps[:-1] if s not in (None, "passthrough")]

    # --- the raw columns
    avail = {}
    need_array = set()
    for fd in (features or []):
        for p, _ in fd.params:
            need_array.add(p)
    for c in schema:
        if c.name in need_array:
            v = _ident(c.name)
            if c.kind == "float":
                w.pre.append("%s = zeros(rows)" % v)
            else:
                w.pre.append("%s = int_zeros(rows)" % v)
    if need_array:
        lines = []
        for c in schema:
            if c.name in need_array:
                v = _ident(c.name)
                lines.append("    %s[i] = %s" % (v, "xf[i, %d]" % c.slot if c.kind == "float" else "xi[i * NI + %d]" % c.slot))
        w.pre.append("for i to rows {\n%s\n}" % "\n".join(lines))
    for c in schema:
        v = "r_" + _ident(c.name)
        if c.kind == "float":
            w.row.append("%s = xf[i, %d]" % (v, c.slot))
            rn, ri = nan_policy.get(c.name, (False, False))
            if rn:
                w.row.append('if %s != %s {\n    stop(f"row {i}: column %s is NaN, which this pipeline refuses (scikit-learn raises ValueError)")\n}' % (v, v, c.name))
            if ri:
                w.row.append('if abs(%s) == inf {\n    stop(f"row {i}: column %s is infinite, which this pipeline refuses (scikit-learn raises ValueError)")\n}' % (v, c.name))
            avail[c.name] = ("num", v)
        elif c.kind == "int":
            w.row.append("%s: Int = xi[i * NI + %d]" % (v, c.slot))
            avail[c.name] = ("int", v)
        else:
            w.row.append("%s: Int = xi[i * NI + %d]" % (v, c.slot))
            avail[c.name] = ("code", v, c)

    # --- the features module
    for n, fd in enumerate(features or []):
        f = "f%d" % n
        args = ", ".join(_ident(p) for p, _ in fd.params)
        w.pre.append("#: features: %s.%s(%s)" % (feature_module_name, fd.name, args))
        w.pre.append("%s = %s.%s(%s)" % (f, feature_module_name, fd.name, args))
        for oname, okind in fd.outputs:
            acc = "%s.%s" % (f, oname) if fd.record else f
            v = "r_" + _ident(oname)
            w.row.append(("%s: Int = %s[i]" if okind == "int" else "%s = %s[i]") % (v, acc))
            avail[oname] = ("int" if okind == "int" else "num", v)
            # a later feature def may read this output as its parameter
            if oname in need_array and oname not in [c.name for c in schema]:
                raise CompileError("feature output %r used as another feature's input is not supported" % oname)

    # --- the transformers
    if pre and isinstance(pre[0], ColumnTransformer):
        vals, names = _column_transformer(w, pre[0], avail)
        rest = pre[1:]
    else:
        names = list(getattr(pre[0] if pre else final, "feature_names_in_", []))
        if not names:
            names = [c.name for c in schema]
        vals = [avail[n] for n in names]
        rest = pre
    for step in rest:
        if isinstance(step, ColumnTransformer):
            raise CompileError("a ColumnTransformer after the first step is not supported")
        vals, names = _transform(w, step, vals, names)
    nz = len(vals)

    # --- the estimator
    is_clf = hasattr(final, "classes_")
    nout, outs, idx = _estimator(w, final, nz, fused)
    if w.uses_z:
        w.row.append("#: Z holds what the estimator reads")
        for j, v in enumerate(vals):
            w.row.append("Z[i, %d] = %s" % (j, _as_num(w, v)))
    if w.z_used:
        w.row.append("#: the features the trees read")
        for j in sorted(w.z_used):
            w.row.append("z%d = %s" % (j, _as_num(w, vals[j])))
    ncol = nout + (1 if is_clf else 0)
    for j, o in enumerate(outs):
        w.est.append("out[i, %d] = %s" % (j, o))
    if is_clf:
        w.est.append("out[i, %d] = float(%s)" % (nout, idx))

    # --- assemble
    def block(lines, ind):
        pad = " " * ind
        return "\n".join(pad + l.replace("\n", "\n" + pad) for l in lines)

    head = [
        "# %s: a scikit-learn model as Viv, written by `vipyr compile`; do not edit." % title,
        "#",
        "# score(xf, xi, rows) takes `rows` input rows: xf holds the float columns",
        "# (rows x NF, row-major) and xi the integer columns and category codes",
        "# (rows * NI, row-major). It returns rows x %d: %s." % (
            ncol, ("the probability of each class, then the predicted class's index" if is_clf else "the prediction")),
        "#",
        "# float columns (xf): %s" % (", ".join(c.name for c in floats) or "none"),
        "# integer columns (xi): %s" % (", ".join("%s%s" % (c.name, " (category code)" if c.kind == "category" else "")
                                               for c in ints) or "none"),
    ]
    if feature_module_name:
        head.append("")
        head.append("import %s" % feature_module_name)
    src = "\n".join(head) + "\n\n"
    src += "NF: Int = %d\nNI: Int = %d\nNZ: Int = %d\nNOUT: Int = %d\n\n" % (NF, NI, nz, ncol)
    body = []
    body.append("if numel(xf) != rows * NF or len(xi) != rows * NI {")
    body.append('    stop(f"score: {rows} rows need {rows * NF} floats and {rows * NI} integers; got {numel(xf)} and {len(xi)}")')
    body.append("}")
    if NF == 0:
        pass
    body += w.pre
    if w.uses_z:
        body.append("Z = zeros(rows, NZ)")
    body.append("out = zeros(rows, NOUT)")
    if w.updates and w.tree_mode == "branches" and w.n_nodes >= BLOCK_NODES:
        body += _blocked(w, block)
    else:
        est = w.est
        if w.updates:
            est = est[:w.split] + [l for u in w.updates for l in u] + est[w.split:]
        body.append("#: row by row: the transformers, then the estimator")
        body.append("for i to rows {\n%s\n}" % block(w.row + ["#: the estimator"] + est, 4))
    body.append("return out")
    # the defs after the body: the blocked layout adds its own
    src += "\n".join(w.consts) + "\n\n" + "\n".join(w.defs) + "\n"
    src += ("#| The model on `rows` rows: see the top of the file.\n"
            "def score(xf: Arr, xi: [Int], rows: Int) -> Arr {\n%s\n}\n" % block(body, 4))
    classes = [str(c) for c in final.classes_] if is_clf else []
    return Emitted(src, schema, ncol, classes, is_clf, w.notes, feature_module_name)
