# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""infer.py: py2viv's whole-program typing of defs (standard library only).

Python's defs take any type; a Viv def is monomorphic once compiled (C and wasm), and py2viv's
own translation of a body depends on its parameters' Python types: an int and a float compute
and print differently (`x * k` is exact integer arithmetic for two ints, `3` against `3.0`), a
Python list and a NumPy array are different values (`v * 2` repeats one and doubles the other,
`b = v[:]` copies one and views the other), an int array divides and prints as itself.

So the parameters of a def are typed by its call sites (every call, on every pass of the
translator's fixed point, not a guess from the body), and a def whose calls pass arguments of
different types is SPECIALIZED: one Viv def per distinct tuple of argument types (a clone,
`f`, `f__t2`, ...), each translated with its own exact types, each call calling the clone of
its argument types. Return types follow per clone (recursion included: the fixed point runs
until no parameter, return or signature changes). A def is specialized only when that is the
same as Python: a plain module-level def (not a method, not a test, not differentiated by
`grad`, not used as a value, not given a function argument, not one that changes an object
passed to it); otherwise its calls' types are joined as before, and a join that would lose
what Python distinguishes is refused ("called with ... and elsewhere with ...").

What stays refused: a value whose type is decided at run time (an item of a list of ints and
floats indexed by a loop variable is an int or a float: a number "of either type", whose
arithmetic and printing py2viv refuses where they differ), more than MAX_VARIANTS clones, and
every join of two types inside one clone.

Type hints (PEP 484 annotations on parameters, `int`, `float`, `bool`, `complex`,
`list[...]`, `np.ndarray`) are evidence, never a source of types: a call whose argument's type
disagrees with the parameter's hint is refused (Python would run it, but the program's own
statement of what the def takes and what the calls pass disagree, and a translation must not
pick one silently), and a def never called is typed by its hints where the guess from the body
has nothing to go on.
"""

import ast

P = None  # the py2viv module (set by py2viv.py)

MAX_VARIANTS = 8


def _sig_known(sig):
    return all(t is not None and t.kind != "unknown" for t in sig)


class InferMixin:
    """The translator's call-site typing of defs (see the module docstring)."""

    # ------------------------------------------------ which defs may be specialized
    def _infer_scan(self):
        """Mark the defs whose calls may be specialized: plain module-level defs whose name is
        only ever called (never read as a value)."""
        valued = set()
        called = set()
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                called.add(id(n.func))
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and id(n) not in called:
                valued.add(n.id)
        for f in self.funcs.values():
            f.spec_ok = bool(getattr(f, "plain", False)) and not f.is_test and f.name not in valued \
                and not f.node.decorator_list and f.name not in getattr(self, "grad_defs", set())
            f.sigs = []
            f.prev_sigs = []
            f.vmap = {}
            f.variants = [f]
            f.vkeys = [None]

    # ------------------------------------------------ each pass
    def _infer_pass_start(self):
        """Plan each def's clones from the argument types its calls passed on the last pass."""
        self._fresh_pass_start()
        self._taint_start()
        self._pass_no = getattr(self, "_pass_no", 0) + 1
        for f in list(self.funcs.values()):
            if not hasattr(f, "sigs"):
                continue
            f.prev_sigs = f.sigs
            f.sigs = []
            plan = self._plan(f)
            if plan is None:
                f.variants, f.vkeys = [f], [None]
                continue
            f.variants, f.vkeys = [], []
            for k, key in enumerate(plan):
                v = f.vmap.get(tuple(t.key() for t in key))
                if v is None:
                    v = self._clone(f, key)
                    f.vmap[tuple(t.key() for t in key)] = v
                v.prev_params = list(key)
                v.param_types = [None] * len(v.params)
                v.prev_ret = v.ret or v.prev_ret
                v.ret = None
                f.variants.append(v)
                f.vkeys.append(key)

    def _plan(self, f):
        """The argument-type tuples to clone f for, in the order the calls came, or None to keep
        one def (its calls' types joined)."""
        if not getattr(f, "spec_ok", False) or self._threaded(f):
            return None
        seen, out = set(), []
        for sig in f.prev_sigs:
            if not _sig_known(sig):
                continue
            sig = tuple(self._sig_type(t) for t in sig)
            if any(t.kind in ("func", "rng") for t in sig):
                return None  # a def passed to it (funcs.py), a Generator's state: one def
            k = tuple(t.key() for t in sig)
            if k not in seen:
                seen.add(k)
                out.append(sig)
        f.too_many = len(out) > MAX_VARIANTS
        if len(out) <= 1 or f.too_many:
            return None
        return out

    def _clone(self, f, key):
        g = P.FuncInfo(f.node)
        g.name = f.name
        g.params = f.params
        g.defaults = f.defaults
        g.locals = f.locals
        g.globals_decl = f.globals_decl
        g.is_test = f.is_test
        g.base = f
        g.threaded, g.prev_threaded = set(), set()
        g.vret, g.vret_set, g.prev_vret = None, False, None
        n = len(f.vmap) + 1
        g.sid = "%s#%d" % (f.name, n)  # the clone's own variables (py2viv.py, _scope_of)
        names = {getattr(v, "viv", None) for v in f.vmap.values()}
        if P.viv_name(f.name) not in names:
            g.viv = P.viv_name(f.name)
        else:
            taken = set(self.all_names) | set(self.funcs) | names
            j = n
            while "%s__t%d" % (f.name, j) in taken or P.viv_name("%s__t%d" % (f.name, j)) in taken:
                j += 1
            g.viv = P.viv_name("%s__t%d" % (f.name, j))
        return g

    def _variant_of(self, f, types, node):
        """The clone of f a call with these argument types calls (f itself when f is one def)."""
        if not hasattr(f, "sigs"):
            return f
        sig = tuple(self._sig_type(t) for t in types)
        k = tuple(t.key() for t in sig)
        if all(tuple(t.key() for t in s) != k for s in f.sigs):
            f.sigs.append(sig)
        if getattr(f, "too_many", False):
            self.refuse(node, "`%s` called with more than %d different tuples of argument types" % (f.name, MAX_VARIANTS),
                        "py2viv makes one def per tuple of argument types, up to %d; convert the arguments to fewer types." % MAX_VARIANTS)
        if len(f.variants) == 1:
            return f.variants[0]
        for v, key in zip(f.variants, f.vkeys):
            if all(t.kind == "unknown" or t == kt for t, kt in zip(sig, key)):
                return v
        return f.variants[0]  # a new signature: its clone is made on the next pass

    def _sig_type(self, t):
        """An argument's type as a clone's key: itself (maps.py names a dict's map type by the
        map it has become, after dicts that met in one name were made one)."""
        return t

    def _variants(self, f):
        return getattr(f, "variants", None) or [f]

    def _infer_state(self):
        out = []
        for k, f in sorted(self.funcs.items()):
            vs = self._variants(f)
            if len(vs) == 1 and vs[0] is f:
                sigs = tuple(tuple(t.key() for t in s) for s in getattr(f, "sigs", []))
                out.append((k, sigs))
                continue
            out.append((k, tuple((v.sid, tuple(v.param_types), v.ret) for v in vs),
                        tuple(tuple(t.key() for t in s) for s in f.sigs)))
        return tuple(out) + (self._fresh_state(),)

    def _viv_def_name(self, f):
        return getattr(f, "viv", None) or P.viv_name(f.name)

    # ------------------------------------------------ lists passed as new values
    # A def may change a list parameter (append, pop, ...) when every call passes a list no
    # other name holds (a literal, a comprehension, list(...), sorted(...), a slice of a list,
    # a copy, or what a def returns that returns only such lists): the caller cannot see the
    # change, so Viv's `lst = append(lst, x)` rebinding the parameter alone is what Python
    # does. The flags are a greatest fixed point: assumed on the first pass, cleared by any call
    # or return that is not one of these, and the translator's passes run until none changes.

    def _fresh_scopes(self):
        for f in list(self.funcs.values()):
            yield f
            for v in getattr(f, "vmap", {}).values():
                yield v

    def _fresh_pass_start(self):
        for v in self._fresh_scopes():
            v.fresh_prev = getattr(v, "fresh_cur", None)
            v.fresh_cur = [True] * len(v.params)
            v.rfresh_prev = getattr(v, "rfresh_cur", None)
            v.rfresh_cur = True
        self._call_variant = {}

    def _fresh_state(self):
        return tuple((getattr(v, "sid", None) or v.name, tuple(getattr(v, "fresh_cur", ())), getattr(v, "rfresh_cur", True))
                     for v in self._fresh_scopes())

    def _param_fresh(self, scope, i):
        prev = getattr(scope, "fresh_prev", None)
        cur = getattr(scope, "fresh_cur", None)
        return (prev is None or prev[i]) and (cur is None or cur[i])

    def _ret_fresh(self, v):
        prev = getattr(v, "rfresh_prev", None)
        return (prev is None or prev) and getattr(v, "rfresh_cur", True)

    def _note_call(self, v, node, i, argnode, t):
        """A call of v (a def or its clone) passes argnode (None: the default) for parameter i."""
        if not hasattr(v, "fresh_cur"):
            return
        if argnode is None or not self._fresh_value(argnode, t, as_arg=True):
            v.fresh_cur[i] = False

    def _note_return(self, value, t):
        f = self.scope
        if f is None or not hasattr(f, "rfresh_cur"):
            return
        if value is None or not self._fresh_value(value, t, as_arg=False):
            f.rfresh_cur = False

    def _fresh_value(self, e, t, as_arg):
        """Whether e's value is a new list no other name holds (see above)."""
        if isinstance(e, ast.List):
            return not any(isinstance(x, ast.Starred) for x in e.elts)
        if isinstance(e, ast.ListComp):
            return True
        if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Mult) and isinstance(e.left, ast.List):
            return True
        if isinstance(e, ast.IfExp):
            return self._fresh_value(e.body, None, as_arg) and self._fresh_value(e.orelse, None, as_arg)
        if isinstance(e, ast.Call):
            fn = e.func
            if isinstance(fn, ast.Name) and fn.id in ("list", "sorted") and fn.id not in self.funcs:
                return True
            if isinstance(fn, ast.Name) and fn.id in self.funcs:
                v = self._call_variant.get(id(e))
                return v is not None and self._ret_fresh(v)
            if isinstance(fn, ast.Attribute) and fn.attr == "copy" and not e.args and not e.keywords:
                return True
            return False
        if isinstance(e, ast.Subscript) and isinstance(e.slice, ast.Slice):
            # a slice of a Python list is a new list (of a NumPy array, a view)
            return t is not None and t.is_arr() and t.container == "list"
        if isinstance(e, ast.Name) and not as_arg:
            f = self.scope
            if e.id in f.params:
                # a new list passed in and given back, which the def kept nowhere else
                return self._param_fresh(f, f.params.index(e.id)) and not self._assigns_name(f.node.body, e.id) \
                    and P.grow_refusal(f.node.body, e.id, set(self.funcs), self.np) is None
            if e.id in f.locals and e.id not in f.globals_decl:
                stores = [m for st in f.node.body for m in ast.walk(st)
                          if isinstance(m, ast.Name) and m.id == e.id and isinstance(m.ctx, ast.Store)]
                par = self.parents
                ok = all(P.funcs.fresh_list_binding(m, par.get(m), par) or
                         isinstance(par.get(m), ast.Assign) and isinstance(par.get(m).value, ast.List)
                         for m in stores)
                return bool(stores) and ok and self._grow_refusal(e) is None
        return False

    # ------------------------------------------------ unknown types that come from refusals
    # A statement py2viv refuses binds nothing it can type; a def whose body has a refusal
    # returns nothing it can type; a parameter only ever passed such values has no type. Every
    # use of such a value is a consequence of the refusal (a cascade), not a construct of its
    # own: its refusal says so, and the corpus reports count the construct once. A value whose
    # type is unknown for any other reason (a guess that did not work out) keeps the plain
    # refusal, which is inference's own open problem.

    def _taint_start(self):
        self.taint_prev = getattr(self, "taint_cur", set()) | getattr(self, "taint_prev", set())
        self.taint_cur = set()
        self.refused_scopes_prev = getattr(self, "refused_scopes", set())
        self.refused_scopes = set()
        for v in self._fresh_scopes():
            v.cparams_prev = getattr(v, "cparams_cur", set())
            v.cparams_cur = set()

    def _taint_refused(self, s):
        if not hasattr(self, "taint_cur"):
            self.taint_cur = set()
            self.refused_scopes = set()
        if self.scope is not None:
            self.refused_scopes.add(getattr(self.scope, "sid", None) or self.scope.name)
        stack = [s]
        while stack:
            m = stack.pop()
            if isinstance(m, ast.Name) and isinstance(m.ctx, ast.Store):
                self.taint_cur.add((self._scope_of(m.id), m.id))
            elif isinstance(m, (ast.Import, ast.ImportFrom)):
                for a in m.names:
                    nm = (a.asname or a.name).split(".")[0]
                    self.taint_cur.add((self._scope_of(nm), nm))
            elif isinstance(m, ast.FunctionDef) and m is not s:
                continue
            for c in ast.iter_child_nodes(m):
                if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                    continue
                stack.append(c)

    def _tainted_name(self, name):
        key = (self._scope_of(name), name)
        return key in getattr(self, "taint_cur", ()) or key in getattr(self, "taint_prev", ()) or \
            name in self.refused_mods

    def _note_tainted_args(self, v, types):
        if not hasattr(v, "cparams_cur"):
            return
        for i, t in enumerate(types):
            if t == P.UNK_C:
                v.cparams_cur.add(i)

    def _tainted_param(self, f, i):
        return i in getattr(f, "cparams_cur", ()) or i in getattr(f, "cparams_prev", ())

    def _unknown_ret(self, v):
        sid = getattr(v, "sid", None) or v.name
        if sid in getattr(self, "refused_scopes", ()) or sid in getattr(self, "refused_scopes_prev", ()):
            return P.UNK_C
        return P.UNK

    def _taint_expr(self, node, r):
        """An expression of unknown type with a part whose type is unknown because of a refusal
        (a name its refused statement bound, a call of a def whose body was refused): a
        cascade too."""
        for m in ast.walk(node):
            if isinstance(m, ast.Name) and isinstance(m.ctx, ast.Load) and self._comp_get(m.id) is None:
                t = self.var_type(m.id)
                if t is None and self._tainted_name(m.id) or t == P.UNK_C:
                    return (r[0], P.UNK_C) + tuple(r[2:])
            if isinstance(m, ast.Call) and isinstance(m.func, ast.Name) and m.func.id in self.funcs:
                v = getattr(self, "_call_variant", {}).get(id(m))
                if v is not None and (v.ret or v.prev_ret) is None and self._unknown_ret(v) == P.UNK_C:
                    return (r[0], P.UNK_C) + tuple(r[2:])
        return r

    def _calls_all_refused(self, f):
        """A def the program calls, none of whose calls was translated on the last pass or
        this one: every call is inside a construct py2viv refused (a class it does not
        translate, a refused statement), so its parameters' types are unknown because of
        those refusals."""
        base = getattr(f, "base", f)
        return not getattr(base, "prev_sigs", None) and not getattr(base, "sigs", None) and self._pass_no > 1

    # ------------------------------------------------ type hints: evidence, never a source
    def _hint_of(self, ann):
        """The type a parameter annotation states, as far as py2viv reads one: int, float,
        bool, complex, list[...] of those (nested for rows), np.ndarray / NDArray (a NumPy
        array, dtype float unless stated). None for anything else (a union, Optional, a class,
        a string py2viv does not parse)."""
        if ann is None:
            return None
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            try:
                ann = ast.parse(ann.value, mode="eval").body
            except SyntaxError:
                return None
        if isinstance(ann, ast.Name):
            return {"int": P.NUM_I, "float": P.NUM_F, "bool": P.BOOL, "complex": P.CNUM, "str": P.STR}.get(ann.id)
        name = self._call_name(ann) if isinstance(ann, ast.Attribute) else None
        if name in ("np.ndarray", "numpy.ndarray", "np.typing.NDArray", "npt.NDArray"):
            return P.T("arr", 0, "float", "nd")  # rank 0: not stated
        if isinstance(ann, ast.Subscript):
            base = ann.value
            bname = base.id if isinstance(base, ast.Name) else (self._call_name(base) if isinstance(base, ast.Attribute) else None)
            if bname in ("list", "List"):
                inner = self._hint_of(ann.slice)
                if inner is None:
                    return None
                if inner.kind == "str":
                    return P.STRS  # list[str]: a list of strings (strs.py)
                if inner.kind in ("num", "bool"):
                    return P.T("arr", 1, inner.py if inner.kind == "num" else "bool", "list")
                if inner.kind == "arr" and inner.container == "list":
                    return P.T("arr", inner.rank + 1, inner.py, "list")
                return None
            if bname in ("NDArray", "npt.NDArray", "np.typing.NDArray"):
                sl = ann.slice
                dt = self._call_name(sl) if isinstance(sl, ast.Attribute) else None
                py = {"np.float64": "float", "np.int64": "int", "np.bool_": "bool"}.get(dt)
                return P.T("arr", 0, py, "nd") if py else None
        return None

    def _param_hint(self, f, i):
        node = f.node
        args = node.args.args
        if i >= len(args):
            return None
        if args[i].annotation is not None:
            return self._hint_of(args[i].annotation)
        sig = self._pythran_exports().get(f.name)
        return sig[i] if sig is not None and i < len(sig) else None

    def _pythran_exports(self):
        """`#pythran export f(float64[], int)`: Pythran's statement of a def's argument types,
        read as annotations are (evidence; a def exported with two signatures has none)."""
        ex = getattr(self, "_pyth", None)
        if ex is not None:
            return ex
        import re
        ex, seen = {}, {}
        for line in self.src.splitlines():
            m = re.match(r"\s*#\s*pythran\s+export\s+(\w+)\s*\((.*)\)\s*$", line)
            if not m:
                continue
            name, body = m.group(1), m.group(2)
            parts, depth, cur = [], 0, ""
            for ch in body:
                if ch in "([":
                    depth += 1
                elif ch in ")]":
                    depth -= 1
                if ch == "," and depth == 0:
                    parts.append(cur)
                    cur = ""
                else:
                    cur += ch
            if cur.strip():
                parts.append(cur)
            sig = [self._pythran_type(x.strip()) for x in parts]
            if name in seen and seen[name] != [t.key() if t else None for t in sig]:
                ex[name] = None
                continue
            seen[name] = [t.key() if t else None for t in sig]
            ex[name] = sig
        self._pyth = {k: v for k, v in ex.items() if v is not None}
        return self._pyth

    @staticmethod
    def _pythran_type(x):
        import re
        base = {"float": ("num", "float"), "float64": ("num", "float"), "double": ("num", "float"),
                "int": ("num", "int"), "int64": ("num", "int"), "long": ("num", "int"),
                "complex": ("cnum", "complex"), "complex128": ("cnum", "complex"), "bool": ("bool", "bool")}
        m = re.fullmatch(r"(\w+)((?:\s*\[[:,\s]*\])*)((?:\s+list)*)", x)
        if not m or m.group(1) not in base:
            return None
        kind, py = base[m.group(1)]
        dims = m.group(2)
        rank = sum(max(1, g.count(":")) for g in re.findall(r"\[[:,\s]*\]", dims))
        lists = len(m.group(3).split())
        if rank and lists:
            return None
        if rank:
            return P.T("carr" if kind == "cnum" else "arr", rank, py, "nd")
        if lists:
            if kind == "cnum":
                return None
            return P.T("arr", lists, py, "list")
        return {"num": P.NUM_I if py == "int" else P.NUM_F, "cnum": P.CNUM, "bool": P.BOOL}[kind]

    def _check_hints(self, f, vals, args, n):
        """A call whose argument's type contradicts its parameter's annotation: refused (the
        program states one type and passes another; Python runs it, but which the translation
        should follow is not py2viv's to pick)."""
        for i, (p, (c, t)) in enumerate(zip(f.params, vals)):
            h = self._param_hint(f, i)
            if h is None or t.kind == "unknown":
                continue
            bad = False
            if h.kind == "num" and h.py == "int" and t.kind == "num" and t.py in ("float", "mixed"):
                bad = True
            elif h.kind == "arr" and t.is_arr() and h.container != t.container:
                bad = True
            elif h.kind == "arr" and t.kind in ("num", "bool", "cnum"):
                bad = True
            elif h.kind in ("num", "bool") and h.py in ("int", "float") and t.is_arr():
                bad = True
            elif (h.kind in ("str", "strs") or t.kind in ("str", "strs")) and h.kind != t.kind:
                bad = True  # a string where the annotation says a number, or the other way
            if bad:
                words = {"float": "a float", "mixed": "a number that is an int on some paths and a float on others",
                         "int": "an int"}.get(t.py) if t.kind == "num" else None
                if t.is_arr():
                    words = "a %s" % ("Python list" if t.container == "list" else "NumPy array")
                self.refuse(args.get(p, n), "`%s` called with %s for `%s`, which is annotated `%s`" % (
                    f.name, words or self._kind_words(t), p, ast.unparse(f.node.args.args[i].annotation)),
                    "The annotation and the call disagree; py2viv follows neither silently. Fix one of them.")

    def _guess_by_hint_or_use(self, f, i, rank):
        """A def no call types (never called): its parameter's annotation, or an int where the
        body uses it only as Python requires an int (a range bound, an index), or None."""
        h = self._param_hint(f, i)
        if h is not None:
            if h.kind == "arr" and h.rank == 0:
                return P.T("arr", max(rank, 1), h.py, "nd")
            return h
        name = f.params[i]
        uses = [m for m in ast.walk(f.node) if isinstance(m, ast.Name) and m.id == name and isinstance(m.ctx, ast.Load)]
        if not uses:
            return None
        par = self.parents
        intish = 0
        for m in uses:
            x = m
            ok = False
            while True:
                p = par.get(x)
                if isinstance(p, ast.Call) and isinstance(p.func, ast.Name) and p.func.id == "range" and x in p.args:
                    ok = True
                    break
                if isinstance(p, ast.Subscript) and p.slice is x:
                    ok = True
                    break
                if isinstance(p, ast.BinOp) and isinstance(p.op, (ast.Add, ast.Sub)) and \
                        all(isinstance(y, (ast.Name, ast.Constant)) and (not isinstance(y, ast.Constant) or type(y.value) is int)
                            for y in (p.left, p.right)):
                    x = p
                    continue
                break
            intish += ok
        return P.NUM_I if intish else None
