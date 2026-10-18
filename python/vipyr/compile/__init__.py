# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Compile a fitted scikit-learn pipeline, and the Viv functions that make its
features, into one C99 file that checks itself.

    from vipyr.compile import compile_model, load_model
    report = compile_model(pipe, holdout_df, "churn.c", features="features.viv")
    m = load_model("churn.c")          # builds it once (cached), via ctypes
    m.predict_proba(df)

    vipyr compile model.pkl --data holdout.csv -o churn.c [--features f.viv]

The steps (examples/deploy/README.md has the whole story):

1. emit: the pipeline as readable Viv (`churn.viv`, written beside the C):
   the features module's defs run first, then the transformers row by row,
   then the estimator, each with scikit-learn's arithmetic in its order.
2. `viv export --infer` makes the Viv a C kernel; driver.py adds the schema,
   row-major C entries, a CSV command line and `--selftest`.
3. verify: the C, built as a library, against scikit-learn on every row of
   the held-out data; a few hundred rows and scikit-learn's outputs for them
   are embedded, and `./churn --selftest` compares them again on any machine.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from .. import _core
from .driver import render
from .emit import Column, CompileError, emit, read_features

EXACT = ["-O2", "-ffp-contract=off"]
__all__ = ["compile_model", "load_model", "CompileError", "CompiledModel"]


# ------------------------------------------------------------ the inputs

def _load_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "model" in obj:
        return obj
    return {"model": obj}


def _read_table(path):
    import pandas as pd
    p = str(path)
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    return pd.read_csv(p, float_precision="round_trip")


def _final(model):
    from sklearn.pipeline import Pipeline
    return model.steps[-1][1] if isinstance(model, Pipeline) else model


def _category_encoders(model):
    """{input column: the OneHotEncoder categories that read it}"""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    out = {}
    steps = [s for _, s in model.steps] if isinstance(model, Pipeline) else [model]
    for s in steps:
        if isinstance(s, ColumnTransformer):
            names = list(s.feature_names_in_)
            for name, trans, _ in s.transformers_:
                cols = [names[c] for c in s._transformer_to_input_indices.get(name, [])]
                first = trans.steps[0][1] if isinstance(trans, Pipeline) else trans
                if type(first).__name__ == "OneHotEncoder":
                    for j, c in enumerate(cols):
                        out[c] = list(first.categories_[j])
    return out


def _schema(model, data, feats):
    """The input columns: the features' parameters and the columns the
    pipeline reads that no feature makes, in the data's column order."""
    fo = {o for fd in feats for o, _ in fd.outputs}
    fparams = {}
    for fd in feats:
        for p, k in fd.params:
            fparams[p] = k
    names = list(getattr(model, "feature_names_in_", []))
    if not names:
        raise CompileError("the model was fitted without column names: fit it on a pandas DataFrame")
    need = [n for n in names if n not in fo] + [p for p in fparams if p not in names]
    missing = [n for n in need if n not in data.columns]
    if missing:
        raise CompileError("the data has no column %s" % ", ".join(repr(m) for m in missing))
    enc = _category_encoders(model)
    cols = []
    for n in [c for c in data.columns if c in need]:
        dt = data[n].dtype
        if n in fparams:
            kind = fparams[n]
        elif dt.kind == "f":
            kind = "float"
        elif dt.kind in "iub":
            kind = "int"
        else:
            if n not in enc:
                raise CompileError("column %r holds %s values and no OneHotEncoder reads it" % (n, dt))
            if any(not isinstance(c, str) for c in enc[n]):
                raise CompileError("column %r: the encoder's categories are not all strings" % n)
            kind = "category"
        cols.append(Column(n, kind, enc.get(n) if kind == "category" else None))
    return cols


def _with_features(df, feats, fmod):
    """df with the feature columns, computed by the Viv features module
    (vipyr.load), as they will be computed in the C."""
    import pandas as pd
    out = df.copy()
    for fd in feats:
        args = df[[p for p, _ in fd.params]]
        res = getattr(fmod, fd.name)(args)
        if isinstance(res, pd.DataFrame):
            for o, _ in fd.outputs:
                out[o] = np.asarray(res[o])
        else:
            out[fd.outputs[0][0]] = np.asarray(res)
    return out


def _sk_outputs(model, X):
    """scikit-learn's outputs, rows x NOUT: probabilities then the class's
    index for a classifier, the prediction for a regressor."""
    final = _final(model)
    if hasattr(final, "classes_"):
        proba = np.asarray(model.predict_proba(X), dtype=np.float64)
        pred = np.asarray(model.predict(X))
        idx = np.array([int(np.flatnonzero(final.classes_ == p)[0]) for p in pred], dtype=np.float64)
        return np.column_stack([proba, idx])
    return np.asarray(model.predict(X), dtype=np.float64).reshape(-1, 1)


def _fused(model, X):
    """Whether scikit-learn's compiled gradient-boosting loop fuses
    `raw += lr * value` into one rounding on this machine: both ways are
    computed exactly (fractions) on a few rows and compared with its own."""
    from sklearn.pipeline import Pipeline
    final = _final(model)
    if type(final).__name__ not in ("GradientBoostingClassifier", "GradientBoostingRegressor"):
        return False
    Xt = model[:-1].transform(X) if isinstance(model, Pipeline) and len(model.steps) > 1 else X
    Xt = np.ascontiguousarray(Xt, dtype=np.float64)[:200]     # predict_stages needs C order
    ref = final._raw_predict(Xt.astype(np.float32))
    init = final._raw_predict_init(Xt)
    X32 = Xt.astype(np.float32)
    lr = float(final.learning_rate)
    plain, fused = init.copy(), init.copy()
    for s in range(final.estimators_.shape[0]):
        for k in range(final.estimators_.shape[1]):
            t = final.estimators_[s, k].tree_
            v = t.value[t.apply(X32), 0, 0]
            plain[:, k] = plain[:, k] + lr * v
            fused[:, k] = [float(Fraction(lr) * Fraction(float(a)) + Fraction(float(b))) for a, b in zip(v, fused[:, k])]
    pf, ff = np.sum(plain != ref), np.sum(fused != ref)
    return bool(ff < pf)


def _nan_policy(model, frame, schema, feats):
    """For each float column the pipeline reads directly: does scikit-learn
    refuse a NaN, an infinity? Asked of scikit-learn itself, one row."""
    fparams = {p for fd in feats for p, _ in fd.params}
    pol = {}
    for c in schema:
        if c.kind != "float" or c.name in fparams:
            continue
        res = []
        for bad in (np.nan, np.inf):
            row = frame.iloc[:1].copy()
            row[c.name] = bad
            try:
                _sk_outputs(model, row)
                res.append(False)
            except ValueError:
                res.append(True)
        pol[c.name] = tuple(res)
    return pol


def _cell(c, v):
    if c.kind == "float":
        v = float(v)
        return "" if v != v else ("inf" if v == np.inf else "-inf" if v == -np.inf else v.hex())
    if c.kind == "int":
        return str(int(v))
    s = str(v)
    return '"' + s.replace('"', '""') + '"' if any(ch in s for ch in ',"\n\r') else s


def _csv_lines(df, schema, decimal=False):
    lines = [",".join(c.name for c in schema)]
    cols = [df[c.name].to_numpy() for c in schema]
    for i in range(len(df)):
        cells = []
        for c, col in zip(schema, cols):
            if decimal and c.kind == "float":
                v = float(col[i])
                cells.append("" if v != v else repr(v))
            else:
                cells.append(_cell(c, col[i]))
        lines.append(",".join(cells))
    return lines


# --------------------------------------------------------------- building

def _cc_list():
    """The C compilers to build with: cc (clang here) and a GCC if one is installed."""
    out = [os.environ.get("CC", "cc")]
    for g in ("gcc-15", "gcc-14", "gcc-13", "gcc"):
        p = shutil.which(g)
        if not p:
            continue
        try:
            v = subprocess.run([p, "--version"], capture_output=True, text=True).stdout
        except OSError:
            continue
        if "Free Software Foundation" in v:
            out.append(p)
            break
    return out


def build_cli(c_path, exe, cc="cc", extra=()):
    t0 = time.perf_counter()
    r = subprocess.run([cc] + EXACT + list(extra) + [str(c_path), "-o", str(exe)], capture_output=True, text=True)
    if r.returncode:
        raise CompileError("%s failed on %s:\n%s" % (cc, c_path, r.stderr[-3000:]))
    return time.perf_counter() - t0, r.stderr


class CompiledModel:
    """A compiled model's C file, built as a shared library and called
    through ctypes. Columns are read from a DataFrame by name."""

    def __init__(self, c_path, lib_path, prefix):
        self.path = Path(c_path)
        self.prefix = prefix
        self.lib = ctypes.CDLL(str(lib_path))
        f = getattr(self.lib, prefix + "_schema_json")
        f.restype = ctypes.c_char_p
        self.schema = json.loads(f().decode("utf-8"))
        self.columns = self.schema["columns"]
        self.classes = self.schema["classes"]
        self.n_out = self.schema["outputs"]
        sig = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        self._fn = {}
        for name in ("predict", "predict_proba", "predict_all"):
            fn = getattr(self.lib, "%s_%s" % (prefix, name))
            fn.argtypes = sig
            fn.restype = ctypes.c_int
            self._fn[name] = fn
        # the column plan, made once: a call on one row costs a few microseconds
        self._names = [c["name"] for c in self.columns]
        self._fl = [c["name"] for c in self.columns if c["kind"] == "float"]
        self._it = [(c["name"], {s: k for k, s in enumerate(c["categories"])} if c["kind"] == "category" else None)
                    for c in self.columns if c["kind"] != "float"]

    @staticmethod
    def _column(df, name):
        """A column's values as an array. On a frame made by slicing (one row,
        `df.iloc[i:i + 1]`) `df[name]` builds and caches a Series per column,
        about 7 us each; pandas' _get_column_array reads the block directly.
        It is internal to pandas, so any surprise falls back to df[name]."""
        get = getattr(df, "_get_column_array", None)
        if get is not None:
            k = df.columns.get_loc(name)
            if isinstance(k, (int, np.integer)):
                try:
                    return np.asarray(get(k))
                except Exception:
                    pass
        return df[name].to_numpy()

    def arrays(self, df):
        """xf (rows x NF float64) and xi (rows * NI int64) from the frame's columns."""
        n = len(df)
        cols = df.columns
        missing = [c for c in self._names if c not in cols]
        if missing:
            raise KeyError("missing column(s): %s" % ", ".join(missing))
        xf = np.empty((n, len(self._fl)), dtype=np.float64)
        for j, c in enumerate(self._fl):
            xf[:, j] = self._column(df, c)
        xi = np.empty((n, len(self._it)), dtype=np.int64)
        for j, (c, pos) in enumerate(self._it):
            col = self._column(df, c)
            if pos is not None:
                xi[:, j] = [pos.get(str(v), -1) for v in col]
            else:
                xi[:, j] = col
        return xf, xi

    def _call(self, name, df, width):
        xf, xi = self.arrays(df)
        out = np.empty((len(df), width), dtype=np.float64)
        msg = ctypes.create_string_buffer(1024)
        s = self._fn[name](xf.ctypes.data, xi.ctypes.data, len(df), out.ctypes.data, msg, 1024)
        if s:
            raise ValueError("%s (status %d)" % (msg.value.decode("utf-8", "replace"), s))
        return out

    def predict_all(self, df):
        return self._call("predict_all", df, self.n_out)

    def predict_proba(self, df):
        return self._call("predict_proba", df, len(self.classes))

    def predict(self, df):
        r = self._call("predict", df, 1)[:, 0]
        if self.classes:
            return np.array([self.classes[int(k)] for k in r])
        return r


def load_model(c_path, cc=None):
    """Build the compiled model's C file as a shared library (once; cached
    under __pycache__/vipyr beside it, by the file's SHA-256 and the C
    compiler) and load it."""
    c_path = Path(c_path).resolve()
    prefix = c_path.stem
    cc = cc or os.environ.get("CC", "cc")
    key = hashlib.sha256(c_path.read_bytes() + cc.encode() + b" ".join(x.encode() for x in EXACT)).hexdigest()[:20]
    cache = c_path.parent / "__pycache__" / "vipyr"
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        cache = Path(tempfile.gettempdir()) / "vipyr-models"
        cache.mkdir(parents=True, exist_ok=True)
    lib = cache / ("%s-%s%s" % (prefix, key, _core.lib_ext()))
    if not lib.exists():
        tmp = lib.with_suffix(".tmp%d" % os.getpid())
        r = subprocess.run([cc] + EXACT + ["-shared", "-fPIC", "-D%s_NO_MAIN" % prefix.upper(), str(c_path), "-o", str(tmp)],
                           capture_output=True, text=True)
        if r.returncode:
            raise _core.VivBuildError("vipyr: %s failed on %s:\n%s" % (cc, c_path, r.stderr[-3000:]))
        os.replace(tmp, lib)
    return CompiledModel(c_path, lib, prefix)


# ----------------------------------------------------------------- compile

def compile_model(model, data, out, features=None, check_rows=300, tolerance=None, log=print, build_check=True,
                  tree_code="auto"):
    """Compile `model` (a fitted Pipeline or estimator, or a .pkl path) into
    the C99 file `out`, verified on `data` (a DataFrame or a CSV/parquet path
    of held-out rows holding the input columns).

    features: a .viv module whose defs make feature columns from input
    columns (by parameter name; a record result's fields, or an array result
    named like the def). They run first, in the C too.

    tree_code: trees as "branches" (a def of nested ifs per tree, with the
    thresholds as constants), "tables" (node arrays walked by a loop), or
    "auto": branches up to emit.BRANCH_NODES nodes in all.

    Returns a report dict: the estimator, what was verified and how closely,
    the file's size, the build times and the self-test's result."""
    import pandas as pd
    if isinstance(model, (str, Path)):
        pkl = Path(model)
        b = _load_pickle(pkl)
        model = b["model"]
        if data is None:
            data = b.get("check_data")
        if features is None and b.get("features"):
            features = pkl.parent / b["features"] if not os.path.isabs(b["features"]) else b["features"]
    if data is None:
        raise CompileError("held-out data is needed to verify the compiled model (--data)")
    if not isinstance(data, pd.DataFrame):
        data = _read_table(data)
    out = Path(out).resolve()
    prefix = out.stem
    if not prefix.isidentifier():
        raise CompileError("the output's name %r must be a C identifier" % prefix)
    feats, fmod, fname = [], None, None
    if features:
        features = Path(features).resolve()
        feats = read_features(features.read_text())
        if not feats:
            raise CompileError("%s has no feature def (all parameters Arr or [Int], a result that is an array "
                               "or a record of arrays)" % features)
        fname = features.stem
        fmod = _core.load(str(features))
    schema = _schema(model, data, feats)
    data = data.reset_index(drop=True)
    full = _with_features(data[[c.name for c in schema]], feats, fmod) if feats else data[[c.name for c in schema]]
    final = _final(model)
    t0 = time.perf_counter()
    expect_full = _sk_outputs(model, full)
    t_sklearn = time.perf_counter() - t0
    fused = _fused(model, full)
    policy = _nan_policy(model, full, schema, feats)

    em = emit(model, schema, feats, fname, policy, fused, title=prefix, tree_code=tree_code)
    viv_src = out.with_suffix(".viv")
    viv_src.write_text(em.source)

    # viv export in a scratch directory holding the program and its features module
    work = Path(tempfile.mkdtemp(prefix="vipyr-compile-"))
    try:
        shutil.copy(viv_src, work / viv_src.name)
        if features:
            shutil.copy(features, work / features.name)
        t0 = time.perf_counter()
        r = subprocess.run([_core.compiler(), "export", "--infer", str(work / viv_src.name), "-o", str(work / "out")],
                           capture_output=True, text=True, env=_core._env())
        t_export = time.perf_counter() - t0
        if r.returncode:
            raise CompileError("viv export failed:\n%s" % r.stderr[-4000:])
        kernel_c = (work / "out" / (prefix + ".c")).read_text()
    finally:
        shutil.rmtree(work, ignore_errors=True)

    check = data.iloc[:check_rows].reset_index(drop=True)
    cframe = _with_features(check[[c.name for c in schema]], feats, fmod) if feats else check
    st_expect = _sk_outputs(model, cframe)
    st_lines = _csv_lines(check, schema)

    def write(tol, own):
        out.write_text(kernel_c + render(prefix, out.name, prefix, schema, em.classes, em.n_out, st_lines,
                                         st_expect.reshape(-1), tol, own))

    # first pass: the library against scikit-learn on every held-out row
    write(0.0, None)
    lib = load_model(out)
    t0 = time.perf_counter()
    got_full = lib.predict_all(data)
    t_c = time.perf_counter() - t0
    got_check = lib.predict_all(check)
    diff = np.abs(got_full - expect_full)
    same = (got_full == expect_full) | (np.isnan(got_full) & np.isnan(expect_full))
    exact = bool(same.all())
    n_bad = int((~same).sum())
    worst = float(np.nanmax(np.where(same, 0.0, diff))) if n_bad else 0.0
    a_, b_ = got_full[~same], expect_full[~same]
    ulps = float(np.max(np.abs(a_ - b_) / np.spacing(np.maximum(np.abs(a_), np.abs(b_))))) if n_bad else 0.0
    rows_bad = int((~same).any(axis=1).sum())
    labels_same = bool(np.array_equal(got_full[:, -1], expect_full[:, -1])) if em.is_classifier else None
    tol = 0.0
    if not exact:
        tol = tolerance if tolerance is not None else 4 * worst
        if worst > tol:
            raise CompileError("the compiled model differs from scikit-learn by %.3g (tolerance %.3g) on %d values"
                               % (worst, tol, n_bad))
        write(tol, got_check.reshape(-1))
    report = {
        "c_file": out.name, "viv_file": viv_src.name, "bytes": out.stat().st_size,
        "lines": out.read_text().count("\n"),
        "estimator": type(final).__name__, "notes": em.notes, "fused_multiply_add": fused,
        "rows_verified": len(data), "outputs_per_row": em.n_out,
        "bit_identical": exact, "values_differing": n_bad, "rows_differing": rows_bad, "max_abs_diff": worst, "max_ulps": ulps, "tolerance": tol,
        "labels_identical": labels_same, "check_rows": len(check),
        "export_s": t_export, "sklearn_predict_s": t_sklearn, "c_library_s": t_c,
        "nan_policy": {k: {"refuses_nan": a, "refuses_inf": b} for k, (a, b) in policy.items()},
    }
    if build_check:
        builds = []
        for cc in _cc_list():
            exe = out.with_name(prefix + ("" if cc == _cc_list()[0] else "-" + Path(cc).name))
            secs, warn = build_cli(out, exe, cc)
            st = subprocess.run([str(exe), "--selftest"], capture_output=True, text=True)
            builds.append({"cc": Path(cc).name, "compile_s": secs, "selftest_status": st.returncode,
                           "selftest": (st.stdout + st.stderr).strip(), "warnings": warn.strip()[:2000]})
            # the CSV path: the held-out rows through the command line, decimal text in
            csv = "\n".join(_csv_lines(data, schema, decimal=True)) + "\n"
            p = subprocess.run([str(exe)], input=csv, capture_output=True, text=True)
            if p.returncode:
                raise CompileError("%s: the command line failed on the held-out rows: %s" % (exe, p.stderr[-2000:]))
            builds[-1]["cli_identical_to_library"] = _cli_matches(p.stdout, got_full, em)
            if exe.name != prefix:
                exe.unlink()
        report["builds"] = builds
    log and log(_summary(report))
    return report


def _cli_matches(text, got, em):
    rows = text.strip().split("\n")[1:]
    vals = []
    for r in rows:
        cells = r.split(",")
        if em.is_classifier:
            vals.append([float(x) for x in cells[1:]] + [float(em.classes.index(cells[0]))])
        else:
            vals.append([float(cells[0])])
    a = np.array(vals)
    return bool(a.shape == got.shape and np.array_equal(a, got))


def _summary(r):
    lines = ["compiled %s -> %s (%d bytes, %d lines; Viv source %s)" % (r["estimator"], r["c_file"], r["bytes"], r["lines"], r["viv_file"])]
    lines += ["  " + n for n in r["notes"]]
    if r["bit_identical"]:
        lines.append("  %d held-out rows: every output bit-identical to scikit-learn's" % r["rows_verified"])
    else:
        lines.append("  %d held-out rows: %d values in %d rows differ from scikit-learn's, by at most %.3g (%.0f ulp; "
                     "embedded tolerance %.3g)%s"
                     % (r["rows_verified"], r["values_differing"], r["rows_differing"], r["max_abs_diff"], r["max_ulps"],
                        r["tolerance"],
                        "; predicted labels identical" if r["labels_identical"] else ""))
    for b in r.get("builds", []):
        lines.append("  %s: built in %.2f s; %s; command line %s" % (
            b["cc"], b["compile_s"], b["selftest"].splitlines()[-1] if b["selftest"] else "no output",
            "matches the library" if b["cli_identical_to_library"] else "DIFFERS from the library"))
    return "\n".join(lines)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="vipyr compile", description="Compile a fitted scikit-learn pipeline to one C99 file")
    ap.add_argument("model", help="a pickled fitted Pipeline or estimator (or a dict with 'model', 'check_data', 'features')")
    ap.add_argument("-o", "--output", required=True, help="the C file to write, e.g. churn.c")
    ap.add_argument("--data", help="held-out rows (CSV or parquet) holding the input columns")
    ap.add_argument("--features", help="a .viv module of feature defs that run first")
    ap.add_argument("--check-rows", type=int, default=300, help="rows embedded for --selftest (default 300)")
    ap.add_argument("--trees", choices=["auto", "branches", "tables"], default="auto",
                    help="trees as nested ifs (branches) or node tables; auto picks by size")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    a = ap.parse_args(argv)
    try:
        r = compile_model(a.model, a.data, a.output, features=a.features, check_rows=a.check_rows,
                          log=None if a.json else print, tree_code=a.trees)
    except CompileError as e:
        print("vipyr compile: %s" % e, file=sys.stderr)
        return 1
    if a.json:
        print(json.dumps(r, indent=2))
    return 0 if all(b["selftest_status"] == 0 for b in r.get("builds", [])) else 1
