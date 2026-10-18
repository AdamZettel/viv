# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""maps.py: Python dicts whose keys are computed at run time, as Viv maps (py2viv, strict).

A Viv map holds numbers under number keys in insertion order and is one
handle, changed in place: what a Python dict of numbers is. This module translates

  - the dicts that make one: `{}`, `dict()`, a dict comprehension, `dict(zip(ks, vs))`,
    `collections.defaultdict(int)` (or `float`, or `lambda: 0`), `collections.Counter()` and
    `Counter(xs)`, and a literal with number keys (`{1: 2.5, 3: 4}`) when its variable is also
    used as a map below (a computed key, a new key, `del`, `pop`, ...); a literal read only by
    constant keys stays a record (classes.py);
  - `d[k] = v`, `d[k]` (KeyError as Python raises it), `d[k] += v`, `k in d`, `len(d)`,
    `del d[k]`, `.get(k, d)`, `.pop(k)`, `.pop(k, d)`, `.setdefault(k, v)`, `.update(e)`,
    `.clear()`, `.copy()`, `dict(d)`, `.keys()`, `.values()`;
  - loops over `d`, `d.keys()`, `d.values()` and `d.items()`, in insertion order, over a
    snapshot of the keys. A body that changes the size of the dict stops the run where Python
    raises "dictionary changed size during iteration"; a body that could both add and remove
    keys is refused (Python then tells same-sized changes apart, and a snapshot cannot);
  - comprehensions over a dict, and `list`, `sorted`, `sum`, `min`, `max`, `enumerate`, `zip`
    of one (its keys, as Python iterates a dict);
  - printing, `{1: 2.5, 3: 4.0}`, each key and value as Python writes an int or a float.

Keys and values are numbers or strings, one kind of key and one kind of value per dict, as
a Viv map holds them: `{c: i for i, c in enumerate(vocab)}`,
word counts, `names[8] = "oxygen"`. A dict comprehension whose loops are plain heads and
filters is written as Viv's map comprehension, `{k: v for ...}`.

What is refused: a dict mixing string and number keys (or values), keys or values of
another kind (Bools, lists, arrays, other dicts), `d.get(k)` without a default
(None), `d.items()` outside a loop, `popitem`, dict comparisons, and printing a Counter or a
defaultdict (their repr differs from a dict's).

The key and value types of a map are learned from its stores over the translator's passes,
as a record's field types are (classes.py): one MapInfo per creating expression, named in its
type `T("map", name=...)`.
"""

import ast

P = None  # the py2viv module, set when the scan runs

# A NaN key: Python stores it (and finds it again only through the same float object); a Viv
# map refuses it, so the translation stops where Python would store one.
NAN_KEY = "py2viv: a NaN dict key (Python keeps it, found again only through the same float object; a Viv map holds no NaN key)"

# The helpers the translation calls: plain Viv, emitted once when used. Each stop is Python's
# exception type; the key it names is computed, so diff.py compares the start of the line.
HELPERS = {
    "py_mget": ("KeyError", '''
#| d[k] of a dict: Python's KeyError when k is not a key.
def py_mget(m: Map, k) {
    if not has(m, k) {
        stop("KeyError")
    }
    return m[k]
}'''),
    "py_mpop": ("KeyError", '''
#| d.pop(k) and del d[k]: Python's KeyError when k is not a key.
def py_mpop(m: Map, k) {
    if not has(m, k) {
        stop("KeyError")
    }
    return remove(m, k)
}'''),
    "py_mset": (NAN_KEY, '''
#| d[k] = v with a float key: a NaN key stops (Python stores it, and finds it again only
#| through the same float object).
def py_mset(m: Map, k, v) {
    if k != k {
        stop("%s")
    }
    m[k] = v
    return 0
}''' % NAN_KEY),
    "py_mdefault": (NAN_KEY, '''
#| A defaultdict's d[k]: the default is stored first when k is not a key.
def py_mdefault(m: Map, k, d) {
    if not has(m, k) {
        if k != k {
            stop("%s")
        }
        m[k] = d
    }
    return m[k]
}''' % NAN_KEY),
    "py_msetdefault": (NAN_KEY, '''
#| d.setdefault(k, v): v is stored when k is not a key; the value under k.
def py_msetdefault(m: Map, k, v) {
    if not has(m, k) {
        if k != k {
            stop("%s")
        }
        m[k] = v
    }
    return m[k]
}''' % NAN_KEY),
    "py_mupdate": (None, '''
#| d.update(e) of another dict: e's keys in e's order.
def py_mupdate(m: Map, e: Map) {
    ks = keys(e)
    for i to len(ks) {
        m[ks[i]] = e[ks[i]]
    }
    return 0
}'''),
}

SIZE_CHANGED = "RuntimeError: dictionary changed size during iteration"

# the methods that change a dict's keys
ADDS = {"setdefault", "update"}
REMOVES = {"pop", "popitem", "clear"}

# builtins that iterate their argument: a dict passed to one is its keys
ITERATES = {"list", "sorted", "sum", "min", "max", "any", "all", "tuple", "set", "enumerate", "zip", "reversed"}


class MapInfo:
    """One dict-creating expression translated as a Viv map."""

    def __init__(self, name, kind, node, default=None):
        self.name = name          # the type's name, `pymap<N>`
        self.kind = kind          # "dict", "defaultdict", "Counter"
        self.node = node
        self.default = default    # a defaultdict's default: "int" or "float"
        self.kt = self.vt = None  # the key and value types this pass has stored
        self.prev_kt = self.prev_vt = None

    def key_t(self):
        return self.kt or self.prev_kt

    def val_t(self):
        return self.vt or self.prev_vt


# the translator whose maps `unify` merges (py2viv's module-level join has no translator)
CURRENT = None


def unify(a, b):
    """Two map types that meet in one name (a variable, a parameter, a return): one map type
    from then on, holding both one's keys and values; None (a refusal) when one is a
    defaultdict or a Counter and the other is not."""
    tr = CURRENT
    ia, ib = tr.map_infos.get(a.name), tr.map_infos.get(b.name)
    if ia is None or ib is None:
        return None
    if ia is ib:
        return P.T("map", name=ia.name)
    if ia.kind != ib.kind or ia.default != ib.default:
        return None
    keep, gone = (ia, ib) if ia.name < ib.name else (ib, ia)
    for which in ("kt", "vt", "prev_kt", "prev_vt"):
        x, y = getattr(keep, which), getattr(gone, which)
        j = P.join(x, y) if x is not None and y is not None else (x or y)
        setattr(keep, which, j if j is not None else x)
    for k, v in list(tr.map_infos.items()):
        if v is gone:
            tr.map_infos[k] = keep
    for k, v in list(tr.map_sites.items()):
        if v is gone:
            tr.map_sites[k] = keep
    return P.T("map", name=keep.name)


def _num_const(node):
    """A number constant (not a Bool), with an optional sign: its value, or None."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(node.operand, ast.Constant) \
            and type(node.operand.value) in (int, float):
        return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    return None


def _scopes(tree):
    """The module's statements (outside any def) and each def's body, as lists of nodes of one
    scope each (a nested def or class is its own scope)."""
    out = []

    def collect(body):
        nodes, stack = [], list(body)
        while stack:
            n = stack.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if isinstance(n, ast.ClassDef):
                    for m in n.body:
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            collect(m.body)
                else:
                    collect(n.body)
                continue
            nodes.append(n)
            stack.extend(ast.iter_child_nodes(n))
        out.append(nodes)
    collect(tree.body)
    return out


class MapsMixin:
    """The translator's handling of maps (mixed into py2viv.Translator before ObjectsMixin, so
    each hook here sees a dict first and hands everything else on)."""

    # ================================================================ the scan

    def _objects_scan(self):
        global P
        import sys
        P = sys.modules[type(self).__module__]
        super()._objects_scan()
        self._maps_scan()

    def _maps_scan(self):
        global CURRENT
        CURRENT = self
        self.map_infos = {}          # name -> MapInfo
        self.map_sites = {}          # id(creating node) -> MapInfo
        self._dd_names, self._counter_names, self._od_names = set(), set(), set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom) and node.module == "collections":
                for a in node.names:
                    local = a.asname or a.name
                    {"defaultdict": self._dd_names, "Counter": self._counter_names,
                     "OrderedDict": self._od_names}.get(a.name, set()).add(local)
        for nodes in _scopes(self.tree):
            literals = {}   # name -> [a number-keyed literal bound to it]
            as_map = set()  # names used as a map in this scope
            consts = {}     # name -> the constant keys stored or read
            for n in nodes:
                self._map_site_of(n)
                if isinstance(n, (ast.Assign, ast.AnnAssign)) and isinstance(n.value, ast.Dict):
                    tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
                    ks = [_num_const(k) if k is not None else None for k in n.value.keys]
                    if len(tgts) == 1 and isinstance(tgts[0], ast.Name) and ks and all(k is not None for k in ks):
                        literals.setdefault(tgts[0].id, []).append((n.value, set(ks)))
                x = self._map_use(n)
                if x is not None:
                    name, key = x
                    if key is None:
                        as_map.add(name)
                    else:
                        consts.setdefault(name, set()).add(key)
            for name, lits in literals.items():
                for lit, ks in lits:
                    floats = any(isinstance(k, float) for k in ks)
                    if name in as_map or floats or not consts.get(name, set()) <= ks:
                        self._new_map(lit, "dict")
        # a literal with a float key or a computed key anywhere, and no string key (the record
        # path takes string and int constants only)
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Dict) and id(n) not in self.map_sites and n.keys and all(k is not None for k in n.keys):
                if any(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in n.keys):
                    continue
                if any(not (isinstance(_num_const(k), int) and not isinstance(_num_const(k), bool)) for k in n.keys):
                    self._new_map(n, "dict")

    def _new_map(self, node, kind, default=None):
        k = len(self.map_infos) + 1
        while "pymap%d" % k in self.map_infos:
            k += 1
        info = MapInfo("pymap%d" % k, kind, node, default)
        self.map_infos[info.name] = info
        self.map_sites[id(node)] = info
        return info

    def _map_site_of(self, n):
        """The MapInfo of n when it makes a map: `{}`, a dict comprehension, `dict()`,
        `dict(zip(...))`, a defaultdict or a Counter (registered when first met: a class's
        constructor is translated from a copy of `__init__`), or a literal the scan made one."""
        info = self.map_sites.get(id(n))
        if info is not None:
            return info
        if isinstance(n, ast.Dict) and not n.keys:
            return self._new_map(n, "dict")
        if isinstance(n, ast.DictComp):
            return self._new_map(n, "dict")
        if isinstance(n, ast.Call):
            name = self._map_ctor(n)
            if name == "dict" and not n.keywords and (not n.args or len(n.args) == 1 and self._is_zip(n.args[0])):
                return self._new_map(n, "dict")
            if name == "OrderedDict" and not n.args and not n.keywords:
                return self._new_map(n, "dict")
            if name == "defaultdict":
                return self._new_map(n, "defaultdict", self._dd_default(n))
            if name == "Counter":
                return self._new_map(n, "Counter")
        return None

    def _map_ctor(self, call):
        f = call.func
        if isinstance(f, ast.Name):
            if f.id == "dict":
                return "dict"
            for name, names in (("defaultdict", self._dd_names), ("Counter", self._counter_names), ("OrderedDict", self._od_names)):
                if f.id in names:
                    return name
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in getattr(self, "coll_mods", ()) \
                and f.attr in ("defaultdict", "Counter", "OrderedDict"):
            return f.attr
        return None

    @staticmethod
    def _is_zip(n):
        return isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "zip" and len(n.args) == 2 \
            and not n.keywords

    @staticmethod
    def _dd_default(call):
        """A defaultdict's default factory: "int", "float", or None (anything else, refused)."""
        if len(call.args) != 1 or call.keywords:
            return None
        a = call.args[0]
        if isinstance(a, ast.Name) and a.id in ("int", "float"):
            return a.id
        if isinstance(a, ast.Lambda) and not a.args.args and isinstance(a.body, ast.Constant) \
                and type(a.body.value) in (int, float) and a.body.value == 0:
            return "int" if type(a.body.value) is int else "float"
        return None

    @staticmethod
    def _map_use(n):
        """(name, None) when n uses a name as a map (a computed key, del, a method that adds or
        removes keys); (name, key) for a constant key; None otherwise."""
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
            k = _num_const(n.slice)
            if isinstance(n.slice, ast.Slice) or isinstance(n.slice, ast.Tuple):
                return None
            if isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
                return None
            return (n.value.id, k)
        if isinstance(n, ast.Delete):
            for t in n.targets:
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                    return (t.value.id, None)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name):
            m = n.func.attr
            if m in ADDS or m in REMOVES or m in REFUSED_METHODS:
                return (n.func.value.id, None)
            if m == "get" and n.args and _num_const(n.args[0]) is None \
                    and not (isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)):
                return (n.func.value.id, None)
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.In, ast.NotIn)) \
                and isinstance(n.comparators[0], ast.Name) and _num_const(n.left) is None \
                and not (isinstance(n.left, ast.Constant) and isinstance(n.left.value, str)):
            return (n.comparators[0].id, None)
        return None

    def _map_param(self, f, name):
        """Whether a def's body uses the parameter `name` as only a dict is used (a dict method,
        `k in name`), so that no array type is guessed for it before a call types it."""
        for n in ast.walk(f.node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) \
                    and n.func.value.id == name and n.func.attr in ("keys", "values", "items", "get", "setdefault", "update"):
                return True
            if isinstance(n, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in n.ops) \
                    and any(isinstance(c, ast.Name) and c.id == name for c in n.comparators):
                return True
        return False

    # ================================================================ maps as arguments
    # A parameter is typed by its calls (infer.py). A dict argument's map type is the call's,
    # and every dict any call passes for one parameter is one map type (as dicts that meet in
    # a variable are): the maps are unified here, before the call picks its clone, so clones
    # are made per kind of argument (a dict at one site, a list or a number at another), never
    # per dict. A dict and a defaultdict (or a Counter) for one parameter are refused, as they
    # are in one variable. An annotation (`dict[int, float]`, `Counter[int]`, `list[int]`) is
    # evidence, as infer.py reads it: an argument that contradicts it is refused.

    def _sig_type(self, t):
        info = self._map_info(t)
        if info is not None:
            return P.T("map", name=info.name)  # the map it has become (unify)
        return super()._sig_type(t)

    def _map_args(self, f, vals, args, n):
        """The argument types of a call of f, each dict's made the parameter's one map type."""
        mp = f.__dict__.setdefault("map_params", {})
        out = []
        for i, (p, (c, t)) in enumerate(zip(f.params, vals)):
            node = args.get(p, n)
            h = self._map_hint(f, i)
            if t.kind == "map":
                old = mp.get(i)
                if old is not None:
                    u = P.join(old, t)
                    if u is None:
                        self.refuse(node, "`%s` called with %s for `%s`, and elsewhere with %s" % (
                            f.name, self._kind_words(t), p, self._kind_words(old)),
                            "One parameter holds one kind of dict: make them the same kind, or write two functions.")
                    t = u
                t = self._sig_type(t)
                mp[i] = t
                if self._param_hint(f, i) is not None:
                    self.refuse(node, "`%s` called with %s for `%s`, which is annotated `%s`" % (
                        f.name, self._kind_words(t), p, ast.unparse(f.node.args.args[i].annotation)),
                        "The annotation and the call disagree; py2viv follows neither silently. Fix one of them.")
                if h is not None:
                    self._map_hint_check(f, i, t, h, node)
            elif h is not None and (t.kind in ("num", "bool", "cnum", "arr", "carr", "rng") or
                                    t.kind == "rec" and any(ci.name == t.name for ci in self.classes.values())):
                self.refuse(node, "`%s` called with %s for `%s`, which is annotated `%s`" % (
                    f.name, self._kind_words(t), p, ast.unparse(f.node.args.args[i].annotation)),
                    "The annotation and the call disagree; py2viv follows neither silently. Fix one of them.")
            out.append((c, t))
        return out

    def _map_hint(self, f, i):
        """A parameter's dict annotation: (kind, key, value), kind "dict", "defaultdict" or
        "Counter", key and value "int", "float", "bool", "other" or None (not stated, Any);
        None when the annotation is not a dict's."""
        args = f.node.args.args
        if i >= len(args) or args[i].annotation is None:
            return None
        ann = args[i].annotation
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            try:
                ann = ast.parse(ann.value, mode="eval").body
            except SyntaxError:
                return None

        def last(x):
            if isinstance(x, ast.Name):
                return x.id
            if isinstance(x, ast.Attribute):
                return x.attr
            return None
        kinds = {"dict": "dict", "Dict": "dict", "Mapping": "dict", "MutableMapping": "dict",
                 "defaultdict": "defaultdict", "DefaultDict": "defaultdict", "Counter": "Counter"}

        def elt(x):
            nm = last(x) if isinstance(x, ast.Name) or isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) \
                and x.value.id in ("typing", "t") else None
            if nm in ("int", "float", "bool"):
                return nm
            if nm == "Any":
                return None
            return "other"
        base = ann.value if isinstance(ann, ast.Subscript) else ann
        kind = kinds.get(last(base))
        if kind is None:
            return None
        if not isinstance(ann, ast.Subscript):
            return (kind, None, "int" if kind == "Counter" else None)
        sl = ann.slice
        if kind == "Counter":
            return (kind, elt(sl), "int")
        if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:
            return (kind, elt(sl.elts[0]), elt(sl.elts[1]))
        return (kind, "other", "other")

    def _map_hint_check(self, f, i, t, h, node):
        info = self._map_info(t)
        kind, hk, hv = h
        ann = ast.unparse(f.node.args.args[i].annotation)
        why = None
        if kind != "dict" and info.kind != kind:
            why = "%s for `%s`, which is annotated `%s`" % (self._kind_words(t), f.params[i], ann)
        for which, hx, tx in (("keys", hk, info.key_t()), ("values", hv, info.val_t())):
            if why is not None or hx is None or tx is None or tx.kind == "unknown":
                continue
            if hx == "other" or hx == "bool" and tx.kind != "bool" or \
                    hx == "int" and tx.kind == "num" and tx.py in ("float", "mixed"):
                words = {"float": "floats", "mixed": "ints on some paths and floats on others",
                         "int": "ints"}.get(tx.py) if tx.kind == "num" else self._kind_words(tx)
                why = "a dict whose %s are %s for `%s`, which is annotated `%s`" % (which, words, f.params[i], ann)
        if why is not None:
            self.refuse(node, "`%s` called with %s" % (f.name, why),
                        "The annotation and the call disagree; py2viv follows neither silently. Fix one of them.")

    # ================================================================ each pass

    def _objects_pass_start(self):
        super()._objects_pass_start()
        for info in self.map_infos.values():
            info.prev_kt = info.kt or info.prev_kt
            info.prev_vt = info.vt or info.prev_vt
            info.kt = info.vt = None

    def _objects_state(self):
        return super()._objects_state() + tuple((k, i.name, i.kt, i.vt) for k, i in sorted(self.map_infos.items()))

    def _map_info(self, t):
        return self.map_infos.get(t.name) if t is not None and t.kind == "map" else None

    def _map_type(self, info):
        return P.T("map", name=info.name)

    def _map_helper(self, name, info=None):
        """The helper `name`, emitted once. A map of strings (a string key or value) has its own
        copy, `py_mget_sn` (string keys, number values) and so on: a Viv def's map parameter has
        one kind of key and one kind of value."""
        exc, text = HELPERS[name]
        kinds = self._map_kinds(info) if info is not None else "nn"
        if kinds != "nn":
            text = text.replace("def %s(" % name, "def %s_%s(" % (name, kinds), 1)
            name = "%s_%s" % (name, kinds)
        if all(n != name for n, _ in self.wrappers):
            self.wrappers.append((name, text.strip("\n")))
            if exc:
                self.stops.append({"line": 0, "message": exc, "exact": False, "prefix": True})
        return name

    def _map_kinds(self, info):
        """`nn`, `sn`, `ns` or `ss`: whether the map's keys, then its values, are strings."""
        k = self._map_num(info, "key")
        v = self._map_num(info, "value")
        return ("s" if k.kind == "str" else "n") + ("s" if v.kind == "str" else "n")

    def _map_learn(self, info, key_t, val_t, node):
        """A store: the key and value types a map holds (numbers, or strings)."""
        if key_t is not None:
            self._map_check(key_t, node, "key")
            self._map_one_kind(info.key_t(), key_t, node, "keys")
            j = P.join(info.kt, key_t) if key_t.kind != "unknown" else info.kt
            info.kt = j
        if val_t is not None:
            self._map_check(val_t, node, "value")
            self._map_one_kind(info.val_t(), val_t, node, "values")
            j = P.join(info.vt, val_t) if val_t.kind != "unknown" else info.vt
            info.vt = j

    def _map_one_kind(self, had, t, node, what):
        """A dict whose keys (or values) are numbers and strings both: refused, as a Viv map has
        one kind of each."""
        if had is not None and t is not None and {had.kind, t.kind} == {"num", "str"}:
            self.refuse(node, "a dict whose %s are both numbers and strings" % what,
                        "A Viv map has one kind of key and one kind of value; "
                        "keep two dicts, or convert with str(...).")

    def _map_check(self, t, node, what):
        if t.kind in ("num", "unknown", "str"):
            return
        words = {"none": "None", "map": "a dict (a map of maps)", "tuple": "a tuple"}.get(t.kind) or self._kind_words(t)
        self.refuse(node, "a dict %s that is %s" % (what, words),
                    "A Viv map holds numbers or strings: an int, a float or a str%s." % (
                        "; a Bool key is 1 or 0 in Python, write int(b)" if t.kind == "bool" and what == "key" else ""))

    def _map_num(self, info, which):
        """The type a key or a value of the map reads as."""
        t = info.key_t() if which == "key" else info.val_t()
        return t if t is not None else P.UNK

    def _map_fix(self, code, t):
        """An int stored into a map: `+ 0.0`, so that an int computed as -0.0 prints `0`."""
        if t is not None and t.kind == "num" and t.py == "int" and _num_const_code(code) is None:
            return self._intfix(code, t)
        return code

    def _map_store(self, base, kc, kt, vc, vt, info=None):
        """The Viv of `base[k] = v`: a key that may be a float goes through py_mset, which stops
        on a NaN."""
        kc, vc = self._map_fix(kc, kt), self._map_fix(vc, vt)
        if kt is not None and (kt.kind == "str" or kt.kind == "num" and kt.py == "int") or _num_const_code(kc) is not None:
            return "%s[%s] = %s" % (base, kc, vc)
        return "%s(%s, %s, %s)" % (self._map_helper("py_mset", info), base, kc, vc)

    def _map_key(self, node):
        """Translate a key: its code and type (a number)."""
        code, t, _ = self.expr(node)
        self._map_check(t, node, "key")
        return code, t

    # ================================================================ making a map

    def _e_Dict(self, n):
        info = self._map_site_of(n)
        if info is None:
            return super()._e_Dict(n)
        if not n.keys:
            return "map()", self._map_type(info), 100
        self._map_hoist_check(n)
        R = self.new_tmp("m")
        lines = ["%s = map()" % R]
        for k, v in zip(n.keys, n.values):
            if k is None:
                self.refuse(n, "`**` in a dict literal", "Write the keys out.")
            kc, kt, _ = self._sub(k, lines, "")
            vc, vt, _ = self._sub(v, lines, "")
            self._map_learn(info, kt, vt, v)
            lines.append(self._map_store(R, kc, kt, vc, vt, info))
        self.pre.extend(lines)
        return R, self._map_type(info), 100

    def _e_DictComp(self, n):
        info = self._map_site_of(n)
        if info is None:
            self._refuse_expr(n)
        self._map_hoist_check(n)
        R = self.new_tmp("m")
        lines, stack = [], []
        env = self._comp_push()
        try:
            ind = self._open_gens(n.generators, n, self._comp_binder(env), lines, "", stack)
            kc, kt, _ = self._sub(n.key, lines, ind)
            vc, vt, _ = self._sub(n.value, lines, ind)
            self._map_learn(info, kt, vt, n)
            lines.append(ind + self._map_store(R, kc, kt, vc, vt, info))
        finally:
            self._comp_pop()
        self._close(lines, ind, stack)
        comp = _viv_map_comprehension(lines, R)
        if comp is not None:
            return comp, self._map_type(info), 100
        self.pre.append("%s = map()" % R)
        self.pre.extend(lines)
        return R, self._map_type(info), 100

    def _map_hoist_check(self, n):
        """A literal or a comprehension is built before the rest of its statement: refused when
        it calls a def and is not the whole value of an assignment, a return or a statement (a
        call outside it that Python makes first would then run second)."""
        if not self._has_effect(n):
            return
        p = self.parents.get(n)
        if isinstance(p, (ast.Assign, ast.AnnAssign, ast.Return, ast.Expr)) and p.value is n:
            return
        self.refuse(n, "a dict built by a call of a def inside a larger expression",
                    "py2viv builds the dict before the rest of the statement; assign it to a variable first.")

    def _map_ctor_call(self, n, info):
        """dict(), dict(zip(ks, vs)), defaultdict(...), Counter(), Counter(xs)."""
        if info.kind == "defaultdict" and info.default is None:
            self.refuse(n, "a defaultdict whose default is not int, float or 0",
                        "A Viv map holds numbers: defaultdict(int) or defaultdict(float).")
        if info.kind in ("dict", "defaultdict") and not n.args or info.kind == "defaultdict":
            return "map()", self._map_type(info), 100
        R = self.new_tmp("m")
        lines = ["%s = map()" % R]
        if info.kind == "Counter":
            if len(n.args) != 1 or n.keywords or self._map_info(self._map_base_type(n.args[0])) is not None:
                self.refuse(n, "Counter with these arguments", "Count an array or a list: Counter(xs).")
            src = self._source(n.args[0], lines, "", n)
            i = self.new_tmp("i")
            lines.append("for %s to %s {" % (i, src[-1]))
            code, t = self._source_item(src, i, n, None)
            self._map_learn(info, t, P.NUM_I, n)
            x = self.new_tmp("x")
            lines.append("    %s = %s" % (x, code))
            lines.append("    " + self._map_store(R, x, t, "get(%s, %s, 0) + 1" % (R, x), P.NUM_I, info))
            lines.append("}")
        else:
            z = n.args[0]
            srcs = [self._source(a, lines, "", n) for a in z.args]
            i = self.new_tmp("i")
            lines.append("for %s to min(%s, %s) {" % (i, srcs[0][-1], srcs[1][-1]))
            (kc, kt), (vc, vt) = [self._source_item(s, i, n, None) for s in srcs]
            self._map_learn(info, kt, vt, n)
            lines.append("    " + self._map_store(R, kc, kt, vc, vt, info))
            lines.append("}")
        self._map_hoist_check(n)
        self.pre.extend(lines)
        return R, self._map_type(info), 100

    # ================================================================ reading and storing

    def _obj_subscript(self, n):
        t = self._map_base_type(n.value)
        info = self._map_info(t)
        if info is None:
            return super()._obj_subscript(n)
        if isinstance(n.slice, (ast.Slice, ast.Tuple)):
            self.refuse(n, "a dict indexed by a slice or a tuple", "A Viv map takes one number key.")
        base, bt, bp = self.expr(n.value)
        kc, kt = self._map_key(n.slice)
        vt = self._map_num(info, "value")
        if info.kind == "Counter":
            return "get(%s, %s, 0)" % (base, kc), (vt if vt.kind != "unknown" else P.NUM_I), 100
        if info.kind == "defaultdict":
            self._map_learn(info, kt, P.NUM_I if info.default == "int" else P.NUM_F, n)
            d = "0" if info.default == "int" else "0.0"
            return "%s(%s, %s, %s)" % (self._map_helper("py_mdefault", info), base, self._map_fix(kc, kt), d), self._map_num(info, "value"), 100
        return "%s(%s, %s)" % (self._map_helper("py_mget", info), base, kc), vt, 100

    def _map_base_type(self, node):
        if isinstance(node, (ast.Name, ast.Attribute)):
            return self._static_type(node)
        return None

    def _obj_subscript_store(self, tgt, code, t, s):
        info = self._map_info(self._map_base_type(tgt.value))
        if info is None:
            return super()._obj_subscript_store(tgt, code, t, s)
        if isinstance(tgt.slice, (ast.Slice, ast.Tuple)):
            self.refuse(tgt, "a dict indexed by a slice or a tuple", "A Viv map takes one number key.")
        self._map_check(t, getattr(s, "value", None) or s, "value")
        if self._has_effect(tgt.slice) and not P.re.fullmatch(r"[A-Za-z_]\w*|-?[0-9.]+", code):
            # Python computes the value before the key: keep that order
            v = self.new_tmp("v")
            self.flush_pre(s)
            self.emit("%s = %s" % (v, code), s)
            code = v
        base, bt, bp = self.expr(tgt.value)
        kc, kt = self._map_key(tgt.slice)
        self._map_learn(info, kt, t, s)
        self.flush_pre(s)
        self.emit(self._map_store(base, kc, kt, code, t, info), s)
        return True

    def _obj_augassign(self, s):
        tgt = s.target
        info = self._map_info(self._map_base_type(tgt.value)) if isinstance(tgt, ast.Subscript) else None
        if info is None:
            return super()._obj_augassign(s)
        if isinstance(tgt.slice, (ast.Slice, ast.Tuple)):
            self.refuse(tgt, "a dict indexed by a slice or a tuple", "A Viv map takes one number key.")
        # d[k] op= v: the key once, then d[k] = d[k] op v, read as Python reads it
        key = tgt.slice
        if not (isinstance(key, ast.Name) or _num_const(key) is not None):
            kc, kt = self._map_key(key)
            tmp = self.new_tmp("k")
            self.set_var(tmp, kt, s)
            self.flush_pre(s)
            self.emit("%s = %s" % (self.vname(tmp), kc), s)
            key = ast.copy_location(ast.Name(id=tmp, ctx=ast.Load()), key)
        read = ast.copy_location(ast.Subscript(value=tgt.value, slice=key, ctx=ast.Load()), tgt)
        value = ast.copy_location(ast.BinOp(left=read, op=s.op, right=s.value), s)
        store = ast.copy_location(ast.Subscript(value=tgt.value, slice=key, ctx=ast.Store()), tgt)
        for x in (read, value, store):
            ast.fix_missing_locations(x)
            self.parents[x] = s
        code, t, _ = self.expr(value)
        self._obj_subscript_store(store, code, t, s)
        return True

    def _s_Delete(self, s, top):
        for tgt in s.targets:
            info = self._map_info(self._map_base_type(tgt.value)) if isinstance(tgt, ast.Subscript) else None
            if info is None:
                self.refuse(tgt, "`del` of %s" % ("an element" if isinstance(tgt, ast.Subscript) else "a name"),
                            "Viv deletes only a dict's key: `del d[k]`.")
            base, _, _ = self.expr(tgt.value)
            kc, kt = self._map_key(tgt.slice)
            self.flush_pre(s)
            self.emit("%s(%s, %s)" % (self._map_helper("py_mpop", info), base, kc), s)

    def _obj_in(self, n, op, right, lc, lt):
        recv = right
        if isinstance(right, ast.Call) and isinstance(right.func, ast.Attribute) and right.func.attr == "keys" \
                and not right.args and not right.keywords:
            recv = right.func.value
        info = self._map_info(self._map_base_type(recv))
        if info is None:
            return super()._obj_in(n, op, right, lc, lt)
        self._map_check(lt, n.left, "key")
        base, _, _ = self.expr(recv)
        code = "has(%s, %s)" % (base, lc)
        return "not %s" % code if isinstance(op, ast.NotIn) else code

    def _obj_builtin(self, n, name):
        if name in ("dict", "OrderedDict", "defaultdict", "Counter") or name.endswith((".defaultdict", ".Counter", ".OrderedDict")):
            info = self._map_site_of(n)
            if info is not None:
                return self._map_ctor_call(n, info)
            if name in ("dict", "OrderedDict") and len(n.args) == 1 and not n.keywords:
                src = self._map_info(self._map_base_type(n.args[0]))
                if src is not None:
                    # a copy that is a plain dict (of a defaultdict or a Counter too)
                    info = self.map_sites.get(("copy", id(n))) or self._new_map(n, "dict")
                    self.map_sites.pop(id(n), None)
                    self.map_sites[("copy", id(n))] = info
                    self._map_learn(info, src.key_t(), src.val_t(), n)
                    base, bt, _ = self.expr(n.args[0])
                    return "copy(%s)" % base, self._map_type(info), 100
        args = [a for a in n.args]
        if args and (name == "len" or name in ITERATES) and self._map_info(self._map_base_type(args[0])) is not None:
            if name == "len":
                if len(args) != 1 or n.keywords:
                    self.refuse(n, "len with these arguments", "Pass one dict.")
                base, _, _ = self.expr(args[0])
                return "len(%s)" % base, P.NUM_I, 100
            # a builtin that iterates a dict iterates its keys
            if any(self._map_body_changes([n])):
                self.refuse(n, "`%s` of a dict with an argument that changes a dict" % name,
                            "Python raises RuntimeError when the dict changes size; compute the keys first.")
            new = ast.copy_location(ast.Call(func=n.func, args=[self._keys_call(a) if self._map_info(self._map_base_type(a)) else a
                                                                 for a in args], keywords=n.keywords), n)
            ast.fix_missing_locations(new)
            self._map_parent(new, n)
            return self._builtin_call(new, name)
        return super()._obj_builtin(n, name)

    def _keys_call(self, node):
        c = ast.copy_location(ast.Call(func=ast.copy_location(ast.Attribute(value=node, attr="keys", ctx=ast.Load()), node),
                                       args=[], keywords=[]), node)
        return ast.fix_missing_locations(c)

    def _map_parent(self, new, old):
        p = self.parents.get(old)
        if p is not None:
            self.parents[new] = p
        for c in ast.iter_child_nodes(new):
            self.parents.setdefault(c, new)
            for g in ast.walk(c):
                for gc in ast.iter_child_nodes(g):
                    self.parents.setdefault(gc, g)

    # ================================================================ methods

    def _obj_method(self, n, base, bt, bp):
        site = self._map_site_of(n)
        if site is not None:
            return self._map_ctor_call(n, site)  # collections.defaultdict(int), collections.Counter(xs)
        info = self._map_info(bt)
        if info is None:
            return super()._obj_method(n, base, bt, bp)
        m = n.func.attr
        args = n.args
        if n.keywords or any(isinstance(a, ast.Starred) for a in args):
            self.refuse(n, "`.%s` with keyword or starred arguments" % m, "Pass the arguments by position.")
        vt = self._map_num(info, "value")
        if m in ("keys", "values") and not args:
            t = self._map_num(info, "key" if m == "keys" else "value")
            if t.kind == "str":
                return "%s(%s)" % (m, base), P.STRS, 100
            py = t.py if t.kind == "num" else "float"
            return "%s(%s)" % (m, base), P.T("arr", 1, py, "list"), 100
        if m == "get" and len(args) == 2:
            kc, kt = self._map_key(args[0])
            dc, dt, _ = self.expr(args[1])
            self._map_check(dt, args[1], "value")
            j = P.join(vt, dt) if vt.kind != "unknown" else dt
            if j is None:
                self.refuse(n, "d.get with a default of another type than the values", "Pass a number default.")
            return "get(%s, %s, %s)" % (base, kc, dc), j, 100
        if m == "get" and len(args) == 1:
            self.refuse(n, "d.get(k) without a default (None when k is not a key)", "`None` is not a Viv value; pass a default.")
        if m == "pop" and len(args) in (1, 2):
            kc, kt = self._map_key(args[0])
            if len(args) == 1:
                return "%s(%s, %s)" % (self._map_helper("py_mpop", info), base, kc), vt, 100
            dc, dt, _ = self.expr(args[1])
            self._map_check(dt, args[1], "value")
            j = P.join(vt, dt) if vt.kind != "unknown" else dt
            if j is None:
                self.refuse(n, "d.pop with a default of another type than the values", "Pass a number default.")
            return "remove(%s, %s, %s)" % (base, kc, dc), j, 100
        if m == "setdefault" and len(args) == 2:
            kc, kt = self._map_key(args[0])
            dc, dt, _ = self.expr(args[1])
            self._map_learn(info, kt, dt, n)
            return "%s(%s, %s, %s)" % (self._map_helper("py_msetdefault", info), base, self._map_fix(kc, kt), self._map_fix(dc, dt)), \
                self._map_num(info, "value"), 100
        if m == "setdefault":
            self.refuse(n, "d.setdefault(k) without a default (None)", "`None` is not a Viv value; pass a default.")
        if m == "update" and len(args) == 1:
            other = self._map_info(self._map_base_type(args[0]))
            if other is None:
                self.refuse(n, "d.update of something other than a dict variable", "Update from a dict: d.update(e).")
            ec, et, _ = self.expr(args[0])
            self._map_learn(info, other.key_t(), other.val_t(), n)
            return "%s(%s, %s)" % (self._map_helper("py_mupdate", info), base, ec), P.NUM_I, 100
        if m == "clear" and not args:
            return "clear(%s)" % base, P.NUM_I, 100
        if m == "copy" and not args:
            return "copy(%s)" % base, bt, 100
        if m == "items":
            self.refuse(n, "d.items() outside a for loop header", "Loop over it: `for k, v in d.items():`.")
        if m in REFUSED_METHODS:
            self.refuse(n, "the dict method `.%s`" % m, REFUSED_METHODS[m])
        self.refuse(n, "the dict method `.%s` with %d arguments" % (m, len(args)), "tools/py2viv/SUBSET.md lists the dict methods translated.")

    def _obj_attr(self, n, base, bt, bp):
        if self._map_info(bt) is not None:
            a = n.attr
            if a.startswith("[") and a.endswith("]"):
                # d["k"] of a map: a constant string key (classes.py writes it as an attribute)
                info = self._map_info(bt)
                key = self._str_lit(ast.literal_eval(a[1:-1]), n)
                self._map_learn(info, P.STR, None, n)
                return "%s(%s, %s)" % (self._map_helper("py_mget", info), base, key), self._map_num(info, "value"), 100
            self.refuse(n, "the attribute `%s` of a dict" % a, "Call a dict method: d.%s(...)." % a)
        return super()._obj_attr(n, base, bt, bp)

    def _obj_store(self, tgt, code, t, s):
        info = self._map_info(self._map_base_type(tgt.value)) if isinstance(tgt.value, (ast.Name, ast.Attribute)) else None
        if info is not None:
            if not (tgt.attr.startswith("[") and tgt.attr.endswith("]")):
                self.refuse(tgt, "an attribute of a dict", "Call a dict method.")
            # d["k"] = v of a map: a constant string key
            self._map_check(t, getattr(s, "value", None) or s, "value")
            key = self._str_lit(ast.literal_eval(tgt.attr[1:-1]), tgt)
            self._map_learn(info, P.STR, t, s)
            base, _, _ = self.expr(tgt.value)
            self.flush_pre(s)
            self.emit(self._map_store(base, key, P.STR, code, t, info), s)
            return True
        return super()._obj_store(tgt, code, t, s)

    def _viv_type(self, t, info, f):
        if t.kind == "map":
            return "Map"
        return super()._viv_type(t, info, f)

    # ================================================================ loops

    def _obj_for(self, s):
        it, tgt = s.iter, s.target
        what, recv = "keys", it
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and it.func.attr in ("keys", "values", "items") \
                and not it.args and not it.keywords:
            recv, what = it.func.value, it.func.attr
        info = self._map_info(self._map_base_type(recv))
        if info is None:
            # enumerate(d), zip(d, ...), reversed(d), sorted(d): its keys
            if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id in ("enumerate", "zip", "reversed") \
                    and any(self._map_info(self._map_base_type(a)) is not None for a in it.args):
                adds, removes = self._map_body_changes(s.body)
                if adds or removes:
                    self.refuse(it, "a loop over %s of a dict whose body changes a dict" % it.func.id,
                                "Python raises RuntimeError when the dict changes size; loop over list(d) instead.")
                new_it = ast.copy_location(ast.Call(func=it.func, args=[self._keys_call(a) if self._map_info(self._map_base_type(a))
                                                                        else a for a in it.args], keywords=it.keywords), it)
                ast.fix_missing_locations(new_it)
                new = ast.copy_location(ast.For(target=tgt, iter=new_it, body=s.body, orelse=s.orelse, type_comment=None), s)
                self._map_parent(new_it, it)
                self.parents[new_it] = new
                self._s_For(new, False)
                return True
            return super()._obj_for(s)
        if s.orelse:
            self.refuse(s, "`for ... else`", "Use a flag variable set before `break`.")
        if what == "items":
            if not (isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2 and all(isinstance(e, ast.Name) for e in tgt.elts)):
                self.refuse(tgt, "this loop target", "Write `for k, v in d.items():`.")
            kname, vname = tgt.elts[0].id, tgt.elts[1].id
        elif isinstance(tgt, ast.Name):
            kname, vname = (tgt.id, None) if what == "keys" else (None, tgt.id)
        else:
            self.refuse(tgt, "this loop target", "Loop over one name.")
        adds, removes = self._map_body_changes(s.body)
        if adds and removes:
            self.refuse(s, "a loop over a dict whose body may both add and remove keys",
                        "Python tells a same-sized change from none (and iterates new keys); loop over list(d) instead.")
        base, bt, _ = self.expr(recv)
        if not P.re.fullmatch(r"[A-Za-z_]\w*", base):
            m = self.new_tmp("d")
            self.flush_pre(s)
            self.emit("%s = %s" % (m, base), s)
            base = m
        ks, n0, i = self.new_tmp("ks"), self.new_tmp("n"), self.new_tmp("i")
        self.flush_pre(s)
        self.emit("%s = keys(%s)" % (ks, base), s)
        checked = adds or removes
        if checked:
            # a change of size stops the run at the next step, as Python's iterator does; the
            # step after the last key only checks
            self.emit("%s = len(%s)" % (n0, base), s)
            self.emit("for %s to len(%s) + 1 {" % (i, ks), s)
            self.indent += 1
            self.stops.append({"line": s.lineno, "message": SIZE_CHANGED, "exact": True})
            self.emit("if len(%s) != %s {" % (base, n0), s)
            self.emit("    stop(%s)" % P.viv_str(SIZE_CHANGED), s)
            self.emit("}", s)
            self.emit("if %s == len(%s) {" % (i, ks), s)
            self.emit("    break", s)
            self.emit("}", s)
        else:
            self.emit("for %s to len(%s) {" % (i, ks), s)
            self.indent += 1
        kt, vt = self._map_num(info, "key"), self._map_num(info, "value")
        key = "%s[%s]" % (ks, i)
        if kname is not None:
            self.set_var(kname, kt, s)
            self.emit("%s = %s" % (self.vname(kname), key), s)
            key = self.vname(kname) if not self._assigns_name(s.body, kname) else key
        if vname is not None:
            if kname is not None and key != self.vname(kname):
                k2 = self.new_tmp("k")
                self.emit("%s = %s" % (k2, key), s)
                key = k2
            self.set_var(vname, vt, s)
            self.emit("%s = %s[%s]" % (self.vname(vname), base, key), s)
        self._block(s.body)
        self.indent -= 1
        self.emit("}", s)
        return True

    def _map_body_changes(self, body):
        """(adds, removes): whether running these statements may add keys to a dict, or remove
        them, directly or through a def they call (any def that stores into a subscript, deletes
        one, or calls a method that changes a dict, counts)."""
        adds = removes = False
        seen = set()
        stack = list(body)
        while stack:
            n = stack.pop()
            for x in ast.walk(n):
                if isinstance(x, ast.Subscript) and isinstance(x.ctx, ast.Store):
                    adds = True
                elif isinstance(x, ast.Delete):
                    removes = True
                elif isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute):
                    if x.func.attr in ADDS:
                        adds = True
                    elif x.func.attr in REMOVES:
                        removes = True
                if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id in self.funcs and x.func.id not in seen:
                    seen.add(x.func.id)
                    stack.extend(self.funcs[x.func.id].node.body)
                if isinstance(x, ast.Subscript) and isinstance(x.ctx, ast.Load) and isinstance(x.value, ast.Name):
                    t = self.var_type(x.value.id)
                    mi = self._map_info(t)
                    if mi is not None and mi.kind == "defaultdict":
                        adds = True  # reading a missing key adds it
        return adds, removes

    # ---------------------------------------------------------------- comprehensions
    def _source(self, e, lines, ind, node):
        recv, what = e, "keys"
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute) and e.func.attr in ("keys", "values") \
                and not e.args and not e.keywords:
            recv, what = e.func.value, e.func.attr
        if isinstance(recv, (ast.Name, ast.Attribute)) and self._map_info(self._map_base_type(recv)) is not None:
            info = self._map_info(self._map_base_type(recv))
            adds, removes = self._map_body_changes([node])
            if adds or removes:
                self.refuse(node, "a comprehension over a dict that changes a dict",
                            "Python raises RuntimeError when the dict changes size; compute the keys first.")
            base, _, _ = self._sub(recv, lines, ind)
            tmp = self.new_tmp("s")
            lines.append(ind + "%s = %s(%s)" % (tmp, what, base))
            t = self._map_num(info, "key" if what == "keys" else "value")
            if t.kind == "str":
                return ("seq", tmp, P.STRS, False, e, "len(%s)" % tmp)
            py = t.py if t.kind == "num" else "float"
            return ("seq", tmp, P.T("arr", 1, py, "list"), False, e, "len(%s)" % tmp)
        return super()._source(e, lines, ind, node)

    def _open_clause(self, it, tgt, node, bind, lines, ind, stack):
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and it.func.attr == "items" and not it.args \
                and not it.keywords and self._map_info(self._map_base_type(it.func.value)) is not None:
            d = it.func.value
            z = ast.copy_location(ast.Call(func=ast.copy_location(ast.Name(id="zip", ctx=ast.Load()), it),
                                           args=[self._keys_call(d), ast.fix_missing_locations(ast.copy_location(
                                               ast.Call(func=ast.copy_location(ast.Attribute(value=d, attr="values", ctx=ast.Load()), it),
                                                        args=[], keywords=[]), it))], keywords=[]), it)
            ast.fix_missing_locations(z)
            self._map_parent(z, it)
            it = z
        return super()._open_clause(it, tgt, node, bind, lines, ind, stack)

    # ================================================================ printing

    def _map_print(self, node, code, t, render, spec, parts, values):
        info = self._map_info(t)
        if info.kind != "dict":
            self.refuse(node, "printing a %s" % info.kind, "Python writes a %s's repr in its own form; print a dict: dict(d)." % info.kind)
        if spec is not None:
            self.refuse(node, "a format spec on a dict", "Python refuses it (TypeError).")
        k = len(values)
        parts.append({"kind": "val", "k": k, "type": {"kind": "map"}, "render": render, "spec": None, "sub": True,
                      "expr": ast.get_source_segment(self.src, node) or ""})
        values.append((code, t, node, k))

    def _map_print_forms(self, node, code, t):
        info = self._map_info(t)
        kt, vt = self._map_num(info, "key"), self._map_num(info, "value")
        specs = []
        for x, what in ((kt, "keys"), (vt, "values")):
            if x.kind == "num" and x.py == "mixed":
                self.refuse(node, "printing a dict whose %s are ints on some paths and floats on others" % what,
                            "Python prints 3 and 3.0 apart; convert them with float(...) when storing.")
            specs.append('"d"' if x.kind == "num" and x.py == "int" else '""')
        return ["fmt(%s, %s, %s)" % (code, specs[0], specs[1])]


def _viv_map_comprehension(lines, R):
    """Viv's map comprehension, `{k: v for ...}`, for the loops that build R, when they are only
    loop heads, `if`s and the one store `R[k] = v`; None otherwise. Viv writes its map
    comprehension as the same loops (lang/viv/src/parser.rs, `map_comprehension`), so the map
    is the same. A head `for i to len(s)` whose counter only reads `x = s[i]` is `for x in s`,
    and with `j = i` first, `for j, x in enumerate(s)`."""
    body = [x.strip() for x in lines]
    clauses, k, opened = [], 0, 0
    used = lambda v, skip: any(P.re.search(r"\b%s\b" % P.re.escape(v), x) for j, x in enumerate(body) if j not in skip)
    while k < len(body):
        m = P.re.fullmatch(r"for (\w+) ((?:to|from) .+) \{", body[k])
        if m:
            v, rest = m.group(1), m.group(2)
            seq = P.re.fullmatch(r"to len\((\w+)\)", rest)
            item = (lambda j: P.re.fullmatch(r"(\w+) = %s\[%s\]" % (P.re.escape(seq.group(1)), P.re.escape(v)), body[j])
                    if seq and j < len(body) else None)
            idx = P.re.fullmatch(r"(\w+) = %s" % P.re.escape(v), body[k + 1]) if k + 1 < len(body) else None
            if seq and idx and item(k + 2) and not used(v, (k, k + 1, k + 2)):
                clauses.append("for %s, %s in enumerate(%s)" % (idx.group(1), item(k + 2).group(1), seq.group(1)))
                k += 3
            elif seq and item(k + 1) and not used(v, (k, k + 1)):
                clauses.append("for %s in %s" % (item(k + 1).group(1), seq.group(1)))
                k += 2
            else:
                clauses.append("for %s %s" % (v, rest))
                k += 1
            opened += 1
            continue
        m = P.re.fullmatch(r"if (.+) \{", body[k])
        if m and clauses:
            clauses.append("if %s" % m.group(1))
            opened += 1
            k += 1
            continue
        break
    if not clauses or not clauses[0].startswith("for ") or k >= len(body) or body[k + 1:] != ["}"] * opened:
        return None
    # the store `R[key] = value`, the key's brackets matched
    st = body[k]
    if not st.startswith(R + "["):
        return None
    depth, j = 0, len(R)
    for j in range(len(R), len(st)):
        depth += {"[": 1, "(": 1, "]": -1, ")": -1}.get(st[j], 0)
        if depth == 0:
            break
    key, tail = st[len(R) + 1:j], st[j + 1:]
    if depth != 0 or not tail.startswith(" = "):
        return None
    value = tail[3:]
    if any(P.re.search(r"\b%s\b" % P.re.escape(R), x) for x in clauses + [key, value]):
        return None
    return "{%s: %s %s}" % (key, value, " ".join(clauses))


def _num_const_code(code):
    return code if P.re.fullmatch(r"-?[0-9]+(\.[0-9]+)?", code) else None


# the dict methods refused, with what to write instead
REFUSED_METHODS = {
    "popitem": "It returns a (key, value) tuple; read list(d)[-1] and pop it.",
    "fromkeys": "Build it with a loop or a comprehension.",
    "most_common": "Sort the keys by their counts yourself.",
    "elements": "Loop over the keys and their counts.",
    "subtract": "Loop over the keys.",
    "total": "sum(c.values())",
    "move_to_end": "Not translated.",
}
