# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""DataFrames in and out of Viv functions (pandas and polars).

A loaded module's function takes a pandas or polars DataFrame in place of
its arrays:

    k = vipyr.load("features.viv")
    out = k.card_features(df)                          # columns by name
    out = k.card_features(df, columns={"ts": "time"})  # param <- column
    d = k.card_features(df, as_frame=False)            # dict of numpy arrays

Binding. A frame passed where a record parameter stands fills the record's
fields from the columns of the same name. A frame passed in any other
position (it must then be the only positional argument) fills every
parameter not given by keyword: an array parameter from the column of its
name, a record parameter field by field. `columns={"name": "column"}` reads
a parameter or field from a column of another name (for a record field,
"field" or "param.field").

Conversions, one rule per column dtype (nothing else is accepted: a column
of another dtype is a TypeError that names the column and the dtype):

    float64, int64     passed as they are. pandas: `Series.to_numpy()`, a view
                       of the frame's block; polars: a view of the Arrow
                       buffer when the column has one chunk and no nulls.
                       Zero copy in both cases.
    other ints/floats  widened to int64 / float64 (a copy).
    bool               int64 0 or 1 (a copy).
    datetime64, any    int64 NANOSECONDS since the Unix epoch, in UTC: a
    unit or time zone  tz-aware column is converted to UTC first, a naive one
                       is taken as it stands. (A copy unless the column is
                       already datetime64[ns], naive.)
    timedelta/duration int64 nanoseconds.
    categorical        int64 codes into the column's categories (a copy, to
                       widen the codes). polars 2 keeps no category list on
                       a Categorical, so its codes index the sorted distinct
                       values; an Enum's codes index its declared categories. The categories are kept, so a result
                       field of the same name comes back as that categorical.
    str / object str   factorized to int64 codes (a copy): pandas in order of
                       first appearance, polars in sorted order. Like a
                       categorical, a result field of the same name is rebuilt.
    nullable           a column holding a null (pandas: NA or NaN; polars: a
                       null) is refused with a ValueError naming it, unless
                       the function declares a parameter or field
                       `<name>_valid: [Int]`; then the values (nulls filled
                       with 0) and the mask (1 valid, 0 null) are both passed.

Results. A record or a tuple of arrays as long as the input becomes a frame
of the input's library: pandas keeps the input's index, polars gets a new
frame. A single array of that length is a Series. Arrays of another length
make a new frame (default index) when they share a length, else the dict.
`as_frame=False` returns the dict of numpy arrays. A result field named like
a datetime, duration, categorical or string input column is converted back.

Errors inside Viv keep their classes (the module's VivError, VivIndexError,
which is also an IndexError, and so on): nothing here catches them.
"""
from __future__ import annotations

import functools

import numpy as np

__all__ = ["is_frame", "frame_library", "column_array", "bind_frame", "wrap_module"]


def frame_library(x):
    """'pandas', 'polars' or None, without importing either library."""
    mod = type(x).__module__ or ""
    if mod.startswith("pandas") and type(x).__name__ == "DataFrame":
        return "pandas"
    if mod.startswith("polars") and type(x).__name__ == "DataFrame":
        return "polars"
    return None


def is_frame(x):
    return frame_library(x) is not None


class _Meta:
    """How a column was converted, so a result of the same name can be
    rebuilt: kind is 'cat', 'str', 'datetime' or 'duration'."""

    def __init__(self, kind, categories=None, tz=None, dtype=None):
        self.kind, self.categories, self.tz, self.dtype = kind, categories, tz, dtype


# ------------------------------------------------------------ pandas columns

def _pandas_column(s, col):
    import pandas as pd
    from pandas.api import types as pt
    dt = s.dtype
    if isinstance(dt, pd.CategoricalDtype):
        codes = s.cat.codes.to_numpy().astype(np.int64)
        return codes, codes < 0, _Meta("cat", categories=s.cat.categories, dtype=dt)
    if pt.is_bool_dtype(dt):
        mask = s.isna().to_numpy()
        vals = s.fillna(False).to_numpy(dtype=bool) if mask.any() else s.to_numpy(dtype=bool)
        return vals.astype(np.int64), mask, None
    if isinstance(dt, pd.DatetimeTZDtype):
        u = s.dt.tz_convert("UTC").dt.tz_localize(None)
        return _pandas_time(u, "datetime64[ns]", _Meta("datetime", tz=dt.tz))
    if pt.is_datetime64_dtype(dt):
        return _pandas_time(s, "datetime64[ns]", _Meta("datetime"))
    if pt.is_timedelta64_dtype(dt):
        return _pandas_time(s, "timedelta64[ns]", _Meta("duration"))
    if pt.is_integer_dtype(dt) or pt.is_float_dtype(dt):
        target = np.int64 if pt.is_integer_dtype(dt) else np.float64
        if isinstance(dt, pd.api.extensions.ExtensionDtype):          # Int64, Float64: masked
            arr = s.array
            mask = np.asarray(arr.isna())
            data = getattr(arr, "_data", None)
            if data is None or data.dtype != target:
                data = arr.to_numpy(dtype=target, na_value=0)
            return data, mask, None
        vals = s.to_numpy()
        if vals.dtype != target:
            vals = vals.astype(target)
        mask = np.isnan(vals) if target is np.float64 else None
        return vals, mask, None
    if pt.is_string_dtype(dt) or pt.is_object_dtype(dt):
        mask = s.isna().to_numpy()
        inferred = pd.api.types.infer_dtype(s, skipna=True)
        if inferred not in ("string", "empty"):
            raise TypeError("column %r has dtype %s (%s values); Viv takes numbers, bools, datetimes, "
                            "durations, categoricals and strings" % (col, dt, inferred))
        codes, uniques = pd.factorize(s)
        return codes.astype(np.int64), mask, _Meta("str", categories=pd.Index(uniques))
    raise TypeError("column %r has dtype %s; Viv takes numbers, bools, datetimes, durations, categoricals "
                    "and strings" % (col, dt))


def _pandas_time(s, unit_dtype, meta):
    mask = s.isna().to_numpy()
    vals = s.to_numpy()
    if vals.dtype != unit_dtype:
        vals = vals.astype(unit_dtype)
    return vals.view(np.int64), mask, meta


# ------------------------------------------------------------ polars columns

def _polars_column(s, col):
    import polars as pl
    dt = s.dtype
    mask = s.is_null().to_numpy() if s.null_count() else None
    if dt in (pl.Int64, pl.Float64):
        if mask is None:
            return s.to_numpy(), None, None                # a view of the Arrow buffer
        return s.fill_null(0).to_numpy(), mask, None
    if dt.is_integer() or dt.is_float():
        target = pl.Int64 if dt.is_integer() else pl.Float64
        return s.cast(target).fill_null(0).to_numpy(), mask, None
    if dt == pl.Boolean:
        return s.cast(pl.Int64).fill_null(0).to_numpy(), mask, None
    if isinstance(dt, pl.Datetime):
        u = s
        if dt.time_zone is not None:
            u = u.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        u = u.cast(pl.Datetime("ns")).to_physical()
        return (u if mask is None else u.fill_null(0)).to_numpy(), mask, _Meta("datetime", tz=dt.time_zone)
    if isinstance(dt, pl.Duration):
        u = s.cast(pl.Duration("ns")).to_physical()
        return (u if mask is None else u.fill_null(0)).to_numpy(), mask, _Meta("duration")
    if isinstance(dt, (pl.Categorical, pl.Enum)):
        # an Enum's categories are its dtype's; a Categorical's (polars 2) are
        # the distinct values present, sorted
        cats = list(dt.categories) if isinstance(dt, pl.Enum) else \
            sorted(s.drop_nulls().cast(pl.String).unique().to_list())
        lookup = {v: i for i, v in enumerate(cats)}
        phys = s.cast(pl.String).replace_strict(lookup, default=-1, return_dtype=pl.Int64).fill_null(0)
        return phys.to_numpy(), mask, _Meta("cat", categories=cats, dtype=dt)
    if dt == pl.String:
        vals = s.fill_null("").to_numpy().astype(str)
        uniques, codes = np.unique(vals, return_inverse=True)
        return codes.astype(np.int64), mask, _Meta("str", categories=list(uniques))
    raise TypeError("column %r has dtype %s; Viv takes numbers, bools, datetimes, durations, categoricals "
                    "and strings" % (col, dt))


def column_array(df, col):
    """The column `col` of a pandas or polars frame as Viv takes it:
    (int64 or float64 array, null mask or None, _Meta or None)."""
    lib = frame_library(df)
    if col not in list(df.columns):
        raise KeyError("the frame has no column %r (it has %s)" % (col, ", ".join(map(str, df.columns))))
    if lib == "pandas":
        vals, mask, meta = _pandas_column(df[col], col)
    else:
        vals, mask, meta = _polars_column(df.get_column(col), col)
    if mask is not None and not mask.any():
        mask = None
    return vals, mask, meta


# ------------------------------------------------------------------ binding

def _signature(mod, kernel):
    """The kernel's parameters as the caller sees them: [(name, fields or
    None)], records folded back from the C arguments."""
    args = mod.KERNELS[kernel][0]
    recs = {first: (param, fields) for param, first, fields in getattr(mod, "RECORD_ARGS", {}).get(kernel, [])}
    out, i = [], 0
    while i < len(args):
        if i in recs:
            param, fields = recs[i]
            out.append((param, list(fields)))
            i += len(fields)
        else:
            out.append((args[i][0], None))
            i += 1
    return out


class _Reader:
    """Reads columns of one frame, remembering what each conversion did."""

    def __init__(self, df, columns, names):
        self.df, self.columns, self.names = df, dict(columns or {}), set(names)
        self.n = len(df)
        self.meta = {}                # result name -> _Meta
        self.masks = {}
        self.arrays = {}              # parameter or field -> array passed (for tests)

    def get(self, name, param=None):
        """The array for parameter or field `name`; `<x>_valid` is the mask
        of column x."""
        col = self.columns.get("%s.%s" % (param, name)) if param else None
        col = col or self.columns.get(name, name)
        if name.endswith("_valid") and col not in list(self.df.columns):
            base = name[:-len("_valid")]
            self.get(base, param) if base not in self.masks else None
            m = self.masks.get(base)
            v = np.ones(self.n, np.int64) if m is None else (~m).astype(np.int64)
            self.arrays[name] = v
            return v
        vals, mask, meta = column_array(self.df, col)
        if mask is not None and (name + "_valid") not in self.names:
            raise ValueError("column %r holds nulls; fill them, or declare `%s_valid: [Int]` beside `%s` "
                             "to receive a validity mask" % (col, name, name))
        self.masks[name] = mask
        if meta is not None:
            self.meta[name] = meta
            self.meta.setdefault(col, meta)
        self.arrays[name] = vals
        return vals


def bind_frame(mod, kernel, args, kw, columns=None):
    """The positional values for `kernel` with every frame replaced by its
    columns, and the frame and _Reader that supplied them (or None)."""
    sig = _signature(mod, kernel)
    names = set()
    for p, fields in sig:
        names.update(fields or [p])
    args = list(args)
    source, reader = None, None
    for i, a in enumerate(args):
        if is_frame(a) and not (i < len(sig) and sig[i][1] is not None):
            if len(args) != 1:
                raise TypeError("%s: a frame that fills the parameters must be the only positional argument; "
                                "give the others by keyword" % kernel)
            source = a
            args = []
    frames = [a for a in args if is_frame(a)]
    first = source if source is not None else (frames[0] if frames else None)
    if first is None:
        return None, None, None
    reader = _Reader(first, columns, names)
    vals = []
    for i, (p, fields) in enumerate(sig):
        if i < len(args):
            v = args[i]
            if is_frame(v):
                if v is not first:
                    reader2 = _Reader(v, columns, names)
                    v = {f: reader2.get(f, p) for f in fields}
                else:
                    v = {f: reader.get(f, p) for f in fields}
        elif p in kw:
            v = kw.pop(p)
            if is_frame(v) and fields:
                v = {f: _Reader(v, columns, names).get(f, p) for f in fields}
        elif source is not None:
            v = {f: reader.get(f, p) for f in fields} if fields else reader.get(p)
        else:
            raise TypeError("%s: missing argument %r" % (kernel, p))
        vals.append(v)
    if kw:
        raise TypeError("%s: unexpected keyword arguments %s" % (kernel, ", ".join(sorted(kw))))
    return vals, first, reader


# ------------------------------------------------------------------ results

def _rebuild(name, v, meta, lib):
    m = meta.get(name)
    if m is None or not isinstance(v, np.ndarray) or v.dtype != np.int64:
        return v
    if lib == "pandas":
        import pandas as pd
        if m.kind in ("cat", "str"):
            return pd.Categorical.from_codes(v, categories=m.categories) if m.kind == "cat" \
                else np.asarray(m.categories, dtype=object)[v]
        if m.kind == "datetime":
            t = pd.to_datetime(v.view("datetime64[ns]"))
            return t.tz_localize("UTC").tz_convert(m.tz) if m.tz is not None else t
        if m.kind == "duration":
            return v.view("timedelta64[ns]")
    else:
        import polars as pl
        if m.kind in ("cat", "str"):
            s = pl.Series(name, np.asarray(m.categories, dtype=object)[v].tolist(), dtype=pl.String)
            return s.cast(m.dtype) if m.kind == "cat" else s
        if m.kind == "datetime":
            s = pl.Series(name, v).cast(pl.Datetime("ns"))
            return s.dt.replace_time_zone("UTC").dt.convert_time_zone(m.tz) if m.tz else s
        if m.kind == "duration":
            return pl.Series(name, v).cast(pl.Duration("ns"))
    return v


def to_frame(out, names, df, reader, as_frame=True):
    """A kernel's result as a frame of `df`'s library (see the module's
    docstring)."""
    if isinstance(out, dict):
        cols = dict(out)
    elif isinstance(out, tuple):
        cols = dict(zip(names, out))
    else:
        cols = {names[0]: out}
    if not as_frame:
        return out
    arrays = [v for v in cols.values() if isinstance(v, np.ndarray)]
    if len(arrays) != len(cols) or any(a.ndim != 1 for a in arrays):
        return out
    lens = {len(a) for a in arrays}
    if len(lens) != 1:
        return out
    same = lens == {len(df)}
    lib = frame_library(df)
    cols = {k: _rebuild(k, v, reader.meta, lib) for k, v in cols.items()}
    if lib == "pandas":
        import pandas as pd
        index = df.index if same else None
        if not isinstance(out, (dict, tuple)):
            return pd.Series(cols[names[0]], index=index, name=names[0])
        return pd.DataFrame(cols, index=index)
    import polars as pl
    if not isinstance(out, (dict, tuple)):
        v = cols[names[0]]
        return v if isinstance(v, pl.Series) else pl.Series(names[0], v)
    return pl.DataFrame([v if isinstance(v, pl.Series) else pl.Series(k, v) for k, v in cols.items()])


# ---------------------------------------------------------------- wrapping

def wrap_module(mod):
    """Replace each kernel of a loaded module by a function that also takes
    frames (`columns=`, `as_frame=`); arrays still go straight through."""
    kernels = getattr(mod, "KERNELS", None)
    if not kernels:
        return mod
    for kernel in kernels:
        raw = getattr(mod, kernel, None)
        if raw is None or getattr(raw, "__vipyr_raw__", None) is not None:
            continue

        def make(kernel, raw):
            res_names = [r[0] for r in kernels[kernel][1]]

            @functools.wraps(raw)
            def call(*args, columns=None, as_frame=True, **kw):
                if not any(is_frame(a) for a in list(args) + list(kw.values())):
                    if columns is not None:
                        raise TypeError("%s: columns= needs a DataFrame argument" % kernel)
                    return raw(*args, **kw)
                vals, df, reader = bind_frame(mod, kernel, args, dict(kw), columns)
                out = raw(*vals)
                return to_frame(out, res_names, df, reader, as_frame)
            call.__vipyr_raw__ = raw
            return call
        setattr(mod, kernel, make(kernel, raw))
    return mod
