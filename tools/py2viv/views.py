# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""views.py: where a NumPy view could be observed, for py2viv (standard library only).

NumPy's basic slices (`a[1:3]`, `A[:, j]`), rows (`A[i]`), transposes (`A.T`), the real and
imaginary parts of a complex array, and `reshape`/`ravel` share memory with the array they
come from. Viv's slices, rows and transposes are copies. The two agree exactly as long as
nothing writes into either array while the other is still read, so py2viv translates such a
value only when this module finds no write that could be seen through the shared memory.

The analysis is syntactic and whole-scope, like `grow_refusal` in py2viv.py: it does not
follow the order of statements, so a write anywhere in the scope counts, before or after the
view is taken. It is conservative on purpose: every refusal names the write it found.

  - An alias graph per scope: `b = a` (an alias: Viv shares the array too), `b = a[1:3]`,
    `b = a.T`, `for row in A` (views: Viv copies), `b = f(a)` where `f` may return `a` or a
    view of it. Names joined by any edge form one class.
  - Write sites: a store into an element or slice (`b[i] = v`, `b[1:3] += v`), an augmented
    assignment of a whole array name (`b += v` is in place in NumPy), an in-place method
    (`b.sort()`, `b.fill(v)`), and an argument passed to a def of the program that writes into
    that parameter (found for every def, to a fixed point over the defs it calls).
  - What a def may return: the parameters (and module names) whose array, or a view of it,
    reaches a `return`.
  - Objects (classes.py): an attribute of an object is part of it, so `obj.v[i] = x` writes
    into `obj`, `a = obj.v` joins `a` to `obj`, and `obj.v = a` (or `C(a)`, a constructor that
    keeps its argument) joins `obj` to `a`; a method call `obj.m(x)` is a call of the def
    `C__m(obj, x)`.
"""

import ast

# Methods that change their receiver in place (NumPy arrays and Python lists).
INPLACE_METHODS = {"sort", "fill", "resize", "put", "itemset", "partition", "setfield", "byteswap",
                   "append", "extend", "insert", "pop", "remove", "clear", "reverse"}
# Attributes, functions and methods whose result is a view of their (first) argument.
VIEW_ATTRS = {"T", "real", "imag", "flat", "mT"}
VIEW_FUNCS = {"np.transpose", "np.reshape", "np.ravel", "np.real", "np.imag", "np.squeeze", "np.diagonal",
              "np.swapaxes", "np.moveaxis", "np.atleast_1d", "np.atleast_2d", "np.expand_dims", "np.asarray",
              "np.asanyarray", "np.broadcast_to", "np.flip", "np.fliplr", "np.flipud", "np.rot90"}
VIEW_METHODS = {"transpose", "reshape", "ravel", "squeeze", "view", "swapaxes", "diagonal"}


def _walk_scope(nodes):
    """Every node under these statements, not entering a nested def, lambda or class."""
    stack = list(nodes)
    while stack:
        n = stack.pop()
        yield n
        for c in ast.iter_child_nodes(n):
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            stack.append(c)


class Views:
    def __init__(self, tree, funcs, callname, rank_of=None, fn_flows=None, methods=None):
        """tree: the module; funcs: def name -> FuncInfo (node, params, locals);
        callname: the translator's `_call_name` (np.transpose whatever numpy is called);
        rank_of(scope, name): the rank the translator gave a variable (0 for a number), or None
        when it has none yet (then the name counts as an array); fn_flows: (def, parameter
        index) -> the defs its callers pass there (a call through that parameter calls one of
        them)."""
        self.tree = tree
        self.funcs = funcs
        self.fn_flows = fn_flows or {}
        self.callname = callname
        self.rank_of = rank_of or (lambda scope, name: None)
        self.methods = methods or {}  # method name -> [(def key, takes the object first)] (classes.py)
        self.scope_name = "<module>"
        self._graphs = {}
        self.writes = {}    # def -> (set of param indices written, set of module names written)
        self.returns = {}   # def -> {param index or ('g', name): is_view}
        self._fixpoint()

    # ------------------------------------------------ calls of the program's defs
    def resolve(self, call):
        """[(def key, the call with its arguments as that def takes them)] for a call of a def of
        the program: `f(x)`, or a method call `obj.m(x)` (every class's method of that name, the
        receiver passed first)."""
        f = call.func
        if isinstance(f, ast.Name) and f.id in self.funcs:
            return [(f.id, call)]
        if isinstance(f, ast.Name) and self.called_defs(f.id):
            return [(g, call) for g in self.called_defs(f.id)]  # a call through a function parameter
        if isinstance(f, ast.Attribute) and f.attr in self.methods and not (self.callname(f) or "").startswith(("np.", "math.")):
            out = []
            for key, takes_self in self.methods[f.attr]:
                if key not in self.funcs:
                    continue
                args = ([f.value] if takes_self else []) + list(call.args)
                out.append((key, ast.copy_location(ast.Call(func=ast.Name(id=key, ctx=ast.Load()), args=args,
                                                            keywords=call.keywords), call)))
            return out
        return []

    # ------------------------------------------------ expressions
    def view_step(self, e):
        """(inner expression, is_view) when e's value is e's inner value or shares memory with it:
        a subscript, a view attribute, function or method, or a call of a def that may return its
        argument. None when e is a new array (or not an array)."""
        if isinstance(e, ast.Subscript):
            if self.is_element(e) or self.is_fancy(e):
                return None
            return e.value, True
        if isinstance(e, ast.Attribute) and e.attr in VIEW_ATTRS:
            return e.value, True
        if isinstance(e, ast.Call):
            f = e.func
            name = self.callname(f)
            if name in VIEW_FUNCS and e.args:
                return e.args[0], name not in ("np.asarray", "np.asanyarray")
            if isinstance(f, ast.Attribute) and f.attr in VIEW_METHODS and name not in VIEW_FUNCS \
                    and not (name or "").startswith(("np.", "math.")):
                return f.value, True
        return None

    # NumPy functions whose result is an integer or Bool array (an index array)
    INDEX_FUNCS = {"np.array", "np.asarray", "np.arange", "np.flatnonzero", "np.argsort", "np.isnan", "np.isfinite",
                   "np.isinf", "np.logical_and", "np.logical_or", "np.logical_not"}

    def is_index_array(self, i):
        """A subscript item that is an array by its form (a mask or integer indices): a name of
        an array, a comparison, a list literal, ~, & or | of them, np.nonzero(a)[k], or a
        NumPy function that gives an array. Anything else counts as a number (the view check
        stays on)."""
        if isinstance(i, ast.Name):
            r = self.rank_of(self.scope_name, i.id)
            return r is not None and r > 0
        if isinstance(i, (ast.Compare, ast.List)):
            return True
        if isinstance(i, ast.UnaryOp) and isinstance(i.op, ast.Invert):
            return self.is_index_array(i.operand)
        if isinstance(i, ast.BinOp) and isinstance(i.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
            return self.is_index_array(i.left) or self.is_index_array(i.right)
        if isinstance(i, ast.Subscript) and isinstance(i.value, ast.Call):
            return self.callname(i.value.func) in ("np.nonzero", "np.where")
        if isinstance(i, ast.Call):
            return self.callname(i.func) in self.INDEX_FUNCS
        return False

    def is_fancy(self, e):
        """a[m], a[idx], A[:, idx], A[I, J]: a subscript with an array among its items gives a
        new array in NumPy (a copy), not a view."""
        sl = e.slice
        items = sl.elts if isinstance(sl, ast.Tuple) else [sl]
        return any(self.is_index_array(i) for i in items)

    def is_element(self, e):
        """a[i], A[i, j], A[i][j]: every axis of a named array indexed by an int (a number, not a
        view). Unknown ranks count as arrays. `None` (np.newaxis) makes a view."""
        k = 0
        node = e
        while isinstance(node, ast.Subscript):
            sl = node.slice
            items = sl.elts if isinstance(sl, ast.Tuple) else [sl]
            if any(isinstance(i, ast.Slice) for i in items):
                return False
            if any(isinstance(i, ast.Constant) and i.value is None or
                   isinstance(i, ast.Attribute) and self.callname(i) == "np.newaxis" for i in items):
                return False
            k += len(items)
            node = node.value
        if not isinstance(node, ast.Name):
            return False
        r = self.rank_of(self.scope_name, node.id)
        return r is not None and k >= r

    def is_array_name(self, name):
        r = self.rank_of(self.scope_name, name)
        return r is None or r > 0

    def roots(self, e, view=False):
        """[(name, is_view)]: the names whose arrays e's value may be, or may share memory with."""
        if isinstance(e, ast.Name):
            return [(e.id, view)]
        if isinstance(e, ast.Starred):
            return self.roots(e.value, view)
        if isinstance(e, ast.IfExp):
            return self.roots(e.body, view) + self.roots(e.orelse, view)
        if isinstance(e, (ast.Tuple, ast.List)):
            out = []
            for x in e.elts:
                out += self.roots(x, view)
            return out
        if isinstance(e, ast.Call) and self.resolve(e):
            out = []
            for key, call in self.resolve(e):
                ret = self.returns.get(key, {})
                params = self.funcs[key].params
                for k, v in ret.items():
                    if isinstance(k, tuple):
                        out.append((k[1], view or v))
                        continue
                    a = self.arg_at(call, k, params)
                    if a is not None:
                        out += self.roots(a, view or v)
            return out
        st = self.view_step(e)
        if st is not None:
            return self.roots(st[0], view or st[1])
        if isinstance(e, ast.Attribute) and e.attr not in ("shape", "size", "ndim", "dtype") \
                and not (self.callname(e) or "").startswith(("np.", "math.")):
            return self.roots(e.value, view)  # an attribute of an object is part of it
        return []

    @staticmethod
    def arg_at(call, k, params):
        if k < len(call.args):
            return call.args[k]
        if k < len(params):
            for kw in call.keywords:
                if kw.arg == params[k]:
                    return kw.value
        return None

    def called_defs(self, name):
        """The defs a call through `name` may call: `name` a parameter of the def being read that
        its callers pass defs to (sorted), else []."""
        f = self.funcs.get(self.scope_name)
        if f is None or name not in f.params or name in self.funcs:
            return []
        return sorted(self.fn_flows.get((f.name, f.params.index(name)), ()))

    # ------------------------------------------------ one scope
    def graph(self, key, body):
        """The alias graph of one scope: name -> {name: is_view} (both directions)."""
        g = self._graphs.get(key)
        if g is not None:
            return g
        g = {}

        def edge(a, b, v):
            if a == b:
                return
            g.setdefault(a, {})
            g.setdefault(b, {})
            g[a][b] = g[a].get(b, False) or v
            g[b][a] = g[b].get(a, False) or v

        def bind(target, value):
            if isinstance(target, ast.Name):
                for r, v in self.roots(value):
                    edge(target.id, r, v)
            elif isinstance(target, (ast.Attribute, ast.Subscript)) and self._obj_root(target) is not None:
                # obj.v = a: the object holds a's array (d["k"] = a likewise)
                for r, v in self.roots(value):
                    edge(self._obj_root(target), r, v)
            elif isinstance(target, (ast.Tuple, ast.List)):
                if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(target.elts):
                    for t, v in zip(target.elts, value.elts):
                        bind(t, v)
                else:
                    for t in target.elts:
                        bind(t, value)
            elif isinstance(target, ast.Starred):
                bind(target.value, value)

        for n in _walk_scope(body):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    bind(t, n.value)
            elif isinstance(n, (ast.AnnAssign, ast.NamedExpr)) and n.value is not None:
                bind(n.target, n.value)
            elif isinstance(n, (ast.For, ast.comprehension)):
                it = n.iter
                seqs = [it]
                if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id in ("enumerate", "zip", "reversed"):
                    seqs = it.args
                    if it.func.id == "enumerate" and isinstance(n.target, ast.Tuple) and len(n.target.elts) == 2:
                        tgts = [n.target.elts[1]]
                    elif it.func.id == "zip" and isinstance(n.target, ast.Tuple):
                        tgts = n.target.elts
                    else:
                        tgts = [n.target]
                else:
                    tgts = [n.target]
                for t, q in zip(tgts, seqs if len(seqs) == len(tgts) else [it] * len(tgts)):
                    for r, v in self.roots(q):
                        for x in ast.walk(t):
                            if isinstance(x, ast.Name):
                                edge(x.id, r, True)  # an element of a matrix is a row: a view
        self._graphs[key] = g
        return g

    def _obj_root(self, t):
        """The name whose object an attribute store `obj.f = v` (or `obj.f.g = v`) writes."""
        if not isinstance(t, ast.Attribute):
            return None
        x = t
        while isinstance(x, ast.Attribute):
            x = x.value
        if isinstance(x, ast.Name) and not (self.callname(t) or "").startswith(("np.", "math.")):
            return x.id
        return None

    @staticmethod
    def component(g, names):
        seen = set(names)
        stack = list(names)
        while stack:
            x = stack.pop()
            for y in g.get(x, {}):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        return seen

    def write_sites(self, body):
        """[(root name, node, how)]: every write into an array in these statements."""
        out = []

        def store(t, node, how):
            for x in ast.walk(t):
                if isinstance(x, ast.Subscript) and isinstance(x.ctx, ast.Store):
                    for r, _ in self.roots(x.value):
                        out.append((r, node, how))
        for n in _walk_scope(body):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    store(t, n, "a store into an element or a slice")
            elif isinstance(n, ast.AnnAssign):
                store(n.target, n, "a store into an element or a slice")
            elif isinstance(n, ast.AugAssign):
                if isinstance(n.target, ast.Name):
                    if self.is_array_name(n.target.id):
                        out.append((n.target.id, n, "`%s=`, which NumPy does in place" % _opsym(n.op)))
                else:
                    store(n.target, n, "a store into an element or a slice")
            elif isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Attribute) and f.attr in INPLACE_METHODS and not (self.callname(f) or "").startswith(("np.", "math.")):
                    for r, _ in self.roots(f.value):
                        out.append((r, n, "`.%s()`, which changes it in place" % f.attr))
                # np.random.shuffle(x) and rng.shuffle(x) shuffle their argument in place
                if isinstance(f, ast.Attribute) and f.attr == "shuffle" and n.args and (
                        self.callname(f) == "np.random.shuffle" or not (self.callname(f) or "").startswith(("np.", "math."))):
                    for r, _ in self.roots(n.args[0]):
                        out.append((r, n, "`%s`, which shuffles it in place" % (self.callname(f) or "shuffle")))
                for key, call in self.resolve(n):
                    params_w, globs_w = self.writes.get(key, (set(), set()))
                    params = self.funcs[key].params
                    for k in params_w:
                        a = self.arg_at(call, k, params)
                        if a is not None:
                            for r, _ in self.roots(a):
                                out.append((r, n, "a call of `%s`, which writes into its argument `%s`" % (key, params[k])))
                    for gname in globs_w:
                        out.append((gname, n, "a call of `%s`, which writes into the module's `%s`" % (key, gname)))
        return out

    # ------------------------------------------------ every def, to a fixed point
    def _fixpoint(self):
        for _ in range(len(self.funcs) + 2):
            changed = False
            for name, f in self.funcs.items():
                body = f.node.body
                self.scope_name = name
                self._graphs.pop(("def", name), None)
                g = self.graph(("def", name), body)
                local = set(f.locals) | set(f.params)
                pw, gw = set(), set()
                for r, _, _ in self.write_sites(body):
                    for m in self.component(g, [r]):
                        if m in f.params:
                            pw.add(f.params.index(m))
                        elif m not in local:
                            gw.add(m)
                ret = {}
                results = [(r, v) for n in _walk_scope(body) if isinstance(n, ast.Return) and n.value is not None
                           for r, v in self.roots(n.value)]
                # a constructor returns the object its fields make (classes.py)
                results += [(r, False) for r in getattr(f, "result_names", [])]
                for r, v in results:
                    for m, mv in self._reach(g, r, v):
                        key = f.params.index(m) if m in f.params else (("g", m) if m not in local else None)
                        if key is not None:
                            ret[key] = ret.get(key, False) or mv
                if (pw, gw) != self.writes.get(name) or ret != self.returns.get(name):
                    self.writes[name] = (pw, gw)
                    self.returns[name] = ret
                    changed = True
            if not changed:
                break
        self.scope_name = "<module>"

    @staticmethod
    def _reach(g, start, view):
        """[(name, is_view)] reachable from start in the alias graph, with whether a view edge
        was crossed."""
        best = {start: view}
        stack = [start]
        while stack:
            x = stack.pop()
            for y, v in g.get(x, {}).items():
                nv = best[x] or v
                if y not in best or (nv and not best[y]):
                    best[y] = nv
                    stack.append(y)
        return list(best.items())

    # ------------------------------------------------ the questions the translator asks
    def scope_body(self, scope):
        """A def's body, or the module's statements outside its defs."""
        self.scope_name = scope.name if scope is not None else "<module>"
        if scope is not None:
            return ("def", scope.name), scope.node.body
        return ("module",), [s for s in self.tree.body if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]

    def observed(self, scope, names, skip=None, at=None):
        """(node, how, name) for the first write in the scope into any array these names may
        share (name is the one written), or into a module array a def called here writes (when
        one of them is a module name or a parameter); None when there is none. skip(name, node,
        how) drops a write the caller refuses elsewhere."""
        key, body = self.scope_body(scope)
        g = self.graph(key, body)
        comp = self.component(g, names)
        if at is not None and at in body:
            # the view is taken by a statement of the scope's own body, outside any loop: the
            # statements before it have run before it exists
            body = body[body.index(at):]
        sites = self.write_sites(body)
        sites.sort(key=lambda s: (getattr(s[1], "lineno", 0), getattr(s[1], "col_offset", 0)))
        for r, node, how in sites:
            if r in comp and not (skip and skip(r, node, how)):
                return node, how, r
        # a def called here that writes a module array, while one of the names may hold it
        module_names = {m for m in comp if scope is None or (m not in scope.locals and m not in scope.params)}
        has_param = scope is not None and any(m in scope.params for m in comp)
        for n in _walk_scope(body):
            if isinstance(n, ast.Call):
                for key, _ in self.resolve(n):
                    gw = self.writes.get(key, (set(), set()))[1]
                    hit = sorted(gw & module_names) or (sorted(gw) if has_param else [])
                    if hit:
                        return n, "a call of `%s`, which writes into the module's `%s`" % (key, hit[0]), hit[0]
        return None

    def writer_call(self, node):
        """The first call of a def of the program under node that writes into an array it was
        passed or a module array, or None."""
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                for key, _ in self.resolve(n):
                    pw, gw = self.writes.get(key, (set(), set()))
                    if pw or gw:
                        return n
        return None

    def aliased(self, scope, name):
        """Whether another name, a caller or a view may hold the array `name` holds: a parameter,
        or a name joined to another in the alias graph."""
        if scope is not None and name in scope.params:
            return True
        key, body = self.scope_body(scope)
        g = self.graph(key, body)
        return len(self.component(g, [name])) > 1


def _opsym(op):
    return {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%",
            ast.Pow: "**", ast.MatMult: "@", ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
            ast.LShift: "<<", ast.RShift: ">>"}[type(op)]


# ---------------------------------------------------------------- memory order
#
# NumPy sums (and multiplies) an array in the order of its memory, not of its indices: a
# transpose `A.T` is summed as A is stored, and so is anything computed elementwise from it
# (NumPy's result keeps its operand's memory order). Viv's transpose is a copy in row order.
# So a sum, mean, variance or product of a matrix that may hold a transpose's memory order
# is refused (a transpose itself, `np.sum(A.T)`, is translated as the sum of A).

# NumPy functions whose result is a new C-ordered array whatever their arguments' order.
C_ORDER_FUNCS = {"np.dot", "np.matmul", "np.outer", "np.inner", "np.vdot", "np.cross", "np.kron", "np.zeros", "np.ones",
                 "np.empty", "np.full", "np.eye", "np.identity", "np.linspace", "np.arange", "np.meshgrid",
                 "np.concatenate", "np.stack", "np.vstack", "np.hstack", "np.column_stack", "np.ascontiguousarray",
                 "np.reshape", "np.ravel", "np.diag", "np.trace", "np.linalg.solve", "np.linalg.inv", "np.linalg.cholesky",
                 "np.linalg.norm", "np.linalg.eigvalsh", "np.fft.fft", "np.fft.ifft", "np.sum", "np.mean", "np.std",
                 "np.var", "np.max", "np.min", "np.amax", "np.amin", "np.prod", "np.argmax", "np.argmin"}
C_ORDER_METHODS = {"copy", "reshape", "ravel", "flatten", "tolist", "sum", "mean", "std", "var", "max", "min", "prod",
                   "argmax", "argmin", "dot"}


class Order:
    """Whether an expression's value may be held in a transpose's memory order (views.py)."""

    def __init__(self, views, tree):
        self.v = views
        self.tree = tree
        self.memo = {}
        self._calls = None

    def calls_of(self, fname):
        if self._calls is None:
            self._calls = {}
            scopes = [(None, [s for s in self.tree.body if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))])]
            scopes += [(f, f.node.body) for f in self.v.funcs.values()]
            for f, body in scopes:
                for n in _walk_scope(body):
                    if isinstance(n, ast.Call):
                        for key, call in self.v.resolve(n):
                            self._calls.setdefault(key, []).append((f, call))
        return self._calls.get(fname, [])

    def rank(self, scope, e):
        if isinstance(e, ast.Name):
            return self.v.rank_of(scope.name if scope is not None else "<module>", e.id)
        return None

    def f_order(self, e, scope):
        key = (id(e), id(scope))
        if key in self.memo:
            return self.memo[key]
        self.memo[key] = False  # a cycle through names: no transpose found on it
        r = self._f(e, scope)
        self.memo[key] = r
        return r

    def _f(self, e, scope):
        if isinstance(e, ast.Attribute):
            if e.attr in ("T", "mT"):
                return self.rank(scope, e.value) != 1
            if e.attr in ("real", "imag"):
                return self.f_order(e.value, scope)
            if e.attr in ("shape", "size", "ndim", "dtype") or (self.v.callname(e) or "").startswith(("np.", "math.")):
                return False
            return self.f_order(e.value, scope)  # a field of an object: whatever the object was made from
        if isinstance(e, ast.Subscript):
            return self.f_order(e.value, scope)
        if isinstance(e, ast.BinOp):
            if isinstance(e.op, ast.MatMult):
                return False
            return self.f_order(e.left, scope) or self.f_order(e.right, scope)
        if isinstance(e, ast.UnaryOp):
            return self.f_order(e.operand, scope)
        if isinstance(e, ast.IfExp):
            return self.f_order(e.body, scope) or self.f_order(e.orelse, scope)
        if isinstance(e, ast.Name):
            return self._name(e.id, scope)
        if isinstance(e, ast.Call):
            f = e.func
            name = self.v.callname(f)
            if self.v.resolve(e):
                for key, call in self.v.resolve(e):
                    fi = self.v.funcs[key]
                    if getattr(fi, "ctor_of", None) is not None:
                        # a constructor keeps its arguments in the object's fields
                        if any(self.f_order(a, scope) for a in call.args) or any(self.f_order(k.value, scope) for k in call.keywords):
                            return True
                        continue
                    if any(isinstance(n, ast.Return) and n.value is not None and self.f_order(n.value, fi)
                           for n in _walk_scope(fi.node.body)):
                        return True
            if isinstance(f, ast.Name) and scope is not None and f.id in scope.params and f.id not in self.v.funcs:
                # a call through a function parameter: any def the callers pass it
                k = scope.params.index(f.id)
                for g in sorted(self.v.fn_flows.get((scope.name, k), ())):
                    fi = self.v.funcs[g]
                    if any(isinstance(n, ast.Return) and n.value is not None and self.f_order(n.value, fi)
                           for n in _walk_scope(fi.node.body)):
                        return True
            if isinstance(f, ast.Name) and f.id in self.v.funcs or isinstance(f, ast.Attribute) and self.v.resolve(e):
                return False  # a def or a method of the program: its returns, above
            if name == "np.transpose":
                return not e.args or self.rank(scope, e.args[0]) != 1
            if name in C_ORDER_FUNCS:
                return False
            if isinstance(f, ast.Attribute) and not (name or "").startswith(("np.", "math.")):
                if f.attr == "transpose":
                    return self.rank(scope, f.value) != 1
                if f.attr in C_ORDER_METHODS:
                    return False
                return self.f_order(f.value, scope)
            return any(self.f_order(a, scope) for a in e.args) or any(self.f_order(k.value, scope) for k in e.keywords)
        return False

    def _name(self, name, scope):
        local = scope is not None and (name in scope.locals or name in scope.params)
        if scope is not None and not local:
            return self._name(name, None)
        body = scope.node.body if scope is not None else [s for s in self.tree.body
                                                         if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for n in _walk_scope(body):
            if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
                tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
                if any(isinstance(x, ast.Name) and x.id == name for t in tgts for x in ast.walk(t)):
                    if self.f_order(n.value, scope):
                        return True
            if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) and n.target.id == name:
                if self.f_order(n.value, scope):
                    return True
            if isinstance(n, ast.For) and any(isinstance(x, ast.Name) and x.id == name for x in ast.walk(n.target)):
                if self.f_order(n.iter, scope):
                    return True
        if scope is not None and name in scope.params:
            k = scope.params.index(name)
            for caller, call in self.calls_of(scope.name):
                a = self.v.arg_at(call, k, scope.params)
                if a is not None and self.f_order(a, caller):
                    return True
        return False
