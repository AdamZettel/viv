# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""funcs.py: py2viv's Python functions and lists (standard library only).

Two halves, both used by py2viv.py:

1. `lift(tree)`, a rewrite of the module's syntax tree before translation, in Python's own
   terms, so that the translator sees only module-level defs and calls it already knows:

   - `f = lambda x: e`, bound once in its scope, becomes `def f(x): return e`;
   - a def nested one level inside a module-level def becomes a module-level def
     `outer__inner` whose first parameters are the enclosing def's variables it reads, passed
     at every call. Python reads such a variable when the nested def runs, and nothing can
     rebind it while the call runs (the nested def has no `nonlocal`), so passing its value at
     the call reads what Python reads. A variable that holds an array is passed as the array,
     which Viv shares as Python does;
   - a lambda passed to a def of the program (and a nested def used as a value) becomes a
     module-level def passed by name, when it reads no variable of an enclosing def (Viv's
     compiled engines have no closures);
   - `map(f, xs)` and `filter(f, xs)` read at once (by list, sorted, sum, min, max, any, all,
     next, a for loop or a comprehension) become generator expressions, and `list(<generator>)`
     a list comprehension.

   What it cannot rewrite exactly it leaves, marked with the reason (`node._py2viv_refuse`),
   which the translator reports at the node.

2. `FuncsMixin`, the translator's methods for comprehensions and generator expressions, calls
   of lambdas and of function parameters, key functions, list methods and the other list
   operations (see each method).
"""

import ast
import copy

REFUSE = "_py2viv_refuse"


def mark(node, what, hint):
    """Leave node for the translator to refuse with this message."""
    setattr(node, REFUSE, (what, hint))
    return node


# ---------------------------------------------------------------- scopes


def target_names(t):
    out = []
    for x in ast.walk(t):
        if isinstance(x, ast.Name):
            out.append(x.id)
    return out


def walk_scope(nodes):
    """Every node under these nodes, not entering a nested def, lambda, class or comprehension."""
    stack = list(nodes)
    while stack:
        n = stack.pop()
        yield n
        for c in ast.iter_child_nodes(n):
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
                              ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                # a comprehension's first iterable is evaluated in this scope
                if isinstance(c, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                    stack.append(c.generators[0].iter)
                continue
            stack.append(c)


def bound_names(body, params=()):
    """The names a function body binds (its locals, by Python's rule), with its parameters."""
    out = set(params)
    for n in walk_scope(body):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
    # a nested def or lambda header's defaults are in this scope; walk_scope skips them
    return out


def free_names(node, bound=frozenset()):
    """The names node reads that neither it nor `bound` binds, by Python's scoping: a lambda's
    parameters and a comprehension's targets are local to them (a comprehension's first
    iterable is read outside it)."""
    out = set()

    def visit(n, b):
        if isinstance(n, ast.Name):
            if isinstance(n.ctx, ast.Load) and n.id not in b:
                out.add(n.id)
            return
        if isinstance(n, ast.Lambda):
            a = n.args
            for d in a.defaults + [d for d in a.kw_defaults if d is not None]:
                visit(d, b)
            ps = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
            ps |= {x.arg for x in (a.vararg, a.kwarg) if x is not None}
            visit(n.body, b | ps)
            return
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = n.args
            for d in a.defaults + [d for d in a.kw_defaults if d is not None]:
                visit(d, b)
            ps = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
            ps |= {x.arg for x in (a.vararg, a.kwarg) if x is not None}
            inner = b | bound_names(n.body, ps)
            for s in n.body:
                visit(s, inner)
            return
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            bb = set(b)
            for i, g in enumerate(n.generators):
                visit(g.iter, b if i == 0 else bb)
                bb |= set(target_names(g.target))
                for c in g.ifs:
                    visit(c, bb)
            if isinstance(n, ast.DictComp):
                visit(n.key, bb)
                visit(n.value, bb)
            else:
                visit(n.elt, bb)
            return
        for c in ast.iter_child_nodes(n):
            visit(c, b)

    visit(node, set(bound))
    return out


def is_const_default(d):
    """A default Python evaluates once at the def and the translation at each call: the same
    only for a literal number, Bool or None."""
    if isinstance(d, ast.Constant) and (d.value is None or isinstance(d.value, (int, float, bool))):
        return True
    if isinstance(d, ast.UnaryOp) and isinstance(d.op, (ast.USub, ast.UAdd)):
        return is_const_default(d.operand)
    return False


def _parents(tree):
    p = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            p[c] = n
    return p


def _pos(n):
    return (getattr(n, "lineno", 0), getattr(n, "col_offset", 0))


# ---------------------------------------------------------------- lists py2viv grows
#
# A list py2viv grows or changes (append, pop, insert, ...) must be one nothing else holds:
# Viv's `a = append(a, x)` gives a new array, so a second name would keep the old one.
# py2viv.py's grow_refusal proves it; these are the forms it
# accepts beyond `[]`, a list literal and `.append(x)`.

LIST_MUTATORS = {"append", "pop", "insert", "extend", "remove", "clear", "reverse", "sort"}
LIST_READERS = {"index", "count", "copy"}


def fresh_list_binding(n, p, parent):
    """A binding of the name n (p its parent) to a new list that no other name holds."""
    value = None
    if isinstance(p, ast.Assign) and p.targets == [n]:
        value = p.value
    elif isinstance(p, ast.AnnAssign) and p.target is n:
        value = p.value
    elif isinstance(p, (ast.Tuple, ast.List)):
        a = parent.get(p)
        if isinstance(a, ast.Assign) and a.targets == [p] and isinstance(a.value, (ast.Tuple, ast.List)) \
                and len(a.value.elts) == len(p.elts):
            value = a.value.elts[p.elts.index(n)]
    if value is None:
        return False
    if isinstance(value, (ast.ListComp,)):
        return True
    if isinstance(value, ast.List) and not any(isinstance(e, ast.Starred) for e in value.elts):
        return True
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mult) and isinstance(value.left, ast.List):
        return True
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in ("list", "sorted"):
        return True
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "copy" and not value.args:
        return True
    return False


def list_method_use(p, call, parent):
    """lst.m(...) that py2viv translates for a list it grows: a mutator as a statement (pop
    also as the whole value of an assignment or a return), a reader anywhere."""
    if not (isinstance(call, ast.Call) and call.func is p):
        return False
    if p.attr in LIST_READERS:
        return True
    if p.attr not in LIST_MUTATORS:
        return False
    st = parent.get(call)
    if isinstance(st, ast.Expr):
        return True
    if p.attr == "pop" and (isinstance(st, (ast.Assign, ast.AnnAssign, ast.Return)) and st.value is call):
        return True
    if p.attr == "pop" and isinstance(st, ast.Call) and isinstance(st.func, ast.Attribute) and st.func.attr == "append" \
            and st.args == [call] and not st.keywords and isinstance(st.func.value, ast.Name) and isinstance(parent.get(st), ast.Expr):
        return True  # other.append(lst.pop(i)): the pop, then the append, as Python runs them
    return False


def mutated_in(stmts, name):
    """Whether a list method changes `name` somewhere in these statements."""
    return any(isinstance(m, ast.Call) and isinstance(m.func, ast.Attribute) and m.func.attr in LIST_MUTATORS
               and isinstance(m.func.value, ast.Name) and m.func.value.id == name for st in stmts for m in ast.walk(st))


# ---------------------------------------------------------------- the rewrite


class Lifter:
    """See the module docstring, part 1."""

    CONSUMERS = {"list", "sorted", "sum", "min", "max", "any", "all", "next", "enumerate", "zip"}

    def __init__(self, tree):
        self.tree = tree
        self.names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        self.names |= {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        self.names |= {a.arg for n in ast.walk(tree) if isinstance(n, ast.arguments) for a in n.args + n.posonlyargs}
        self.k = 0

    def fresh(self, stem):
        while True:
            self.k += 1
            cand = "%s%d" % (stem, self.k)
            if cand not in self.names:
                self.names.add(cand)
                return cand

    # the statement lists at module level: the module body, and a main guard's body
    def module_lists(self):
        out = [self.tree.body]
        for s in self.tree.body:
            if isinstance(s, ast.If) and _is_main_guard(s):
                out.append(s.body)
        return out

    def run(self):
        self.map_filter()
        for body in self.module_lists():
            self.lambda_defs(body, None)
        for body in self.module_lists():
            for s in list(body):
                if isinstance(s, ast.FunctionDef):
                    self.lambda_defs(s.body, s)
                    self.lift_nested(s, body)
        self.lambda_args()
        return self.tree

    # ------------------------------------------------ map and filter
    def map_filter(self):
        """`map(f, xs, ...)`, `filter(f, xs)` read at once -> a generator expression;
        `list(<generator>)` -> a list comprehension."""
        par = _parents(self.tree)

        def consumed(call):
            p = par.get(call)
            if isinstance(p, ast.Call) and isinstance(p.func, ast.Name) and p.func.id in self.CONSUMERS and call in p.args:
                return True
            if isinstance(p, ast.Call) and isinstance(p.func, ast.Attribute) and p.func.attr == "join" and p.args == [call]:
                return True  # sep.join(map(str, xs)) (strs.py)
            if isinstance(p, (ast.For, ast.comprehension)) and p.iter is call:
                return True
            return False

        class T(ast.NodeTransformer):
            def visit_Call(tr, n):  # noqa: N805
                tr.generic_visit(n)
                f = n.func
                if isinstance(f, ast.Name) and f.id in ("map", "filter") and not n.keywords and consumed(n) \
                        and not any(isinstance(a, ast.Starred) for a in n.args):
                    if f.id == "map" and len(n.args) >= 2:
                        ts = [self.fresh("_it") for _ in n.args[1:]]
                        call = ast.Call(func=n.args[0], args=[ast.Name(id=t, ctx=ast.Load()) for t in ts], keywords=[])
                        if len(ts) == 1:
                            tgt, it = ast.Name(id=ts[0], ctx=ast.Store()), n.args[1]
                        else:
                            tgt = ast.Tuple(elts=[ast.Name(id=t, ctx=ast.Store()) for t in ts], ctx=ast.Store())
                            it = ast.Call(func=ast.Name(id="zip", ctx=ast.Load()), args=list(n.args[1:]), keywords=[])
                        g = ast.GeneratorExp(elt=call, generators=[ast.comprehension(target=tgt, iter=it, ifs=[], is_async=0)])
                        return _fix(g, n)
                    if f.id == "filter" and len(n.args) == 2:
                        t = self.fresh("_it")
                        fn = n.args[0]
                        cond = ast.Name(id=t, ctx=ast.Load()) if isinstance(fn, ast.Constant) and fn.value is None else \
                            ast.Call(func=fn, args=[ast.Name(id=t, ctx=ast.Load())], keywords=[])
                        g = ast.GeneratorExp(elt=ast.Name(id=t, ctx=ast.Load()),
                                             generators=[ast.comprehension(target=ast.Name(id=t, ctx=ast.Store()), iter=n.args[1],
                                                                           ifs=[cond], is_async=0)])
                        return _fix(g, n)
                if isinstance(f, ast.Name) and f.id == "list" and len(n.args) == 1 and not n.keywords \
                        and isinstance(n.args[0], ast.GeneratorExp):
                    g = n.args[0]
                    return _fix(ast.ListComp(elt=g.elt, generators=g.generators), n)
                return n
        T().visit(self.tree)

    # ------------------------------------------------ f = lambda
    def lambda_defs(self, body, fdef):
        """`f = lambda x: e` bound once in its scope (a module-level list, or a def's body at any
        depth) -> `def f(x): return e` in its place."""
        scope_nodes = body if fdef is None else fdef.body
        stores = {}
        for n in walk_scope(scope_nodes if fdef is not None else self.tree.body):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                stores[n.id] = stores.get(n.id, 0) + 1
            elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                stores[n.name] = stores.get(n.name, 0) + 1
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                for g in n.names:
                    stores[g] = stores.get(g, 0) + 2
        if fdef is None:
            # a def that declares the name global may rebind it
            for n in ast.walk(self.tree):
                if isinstance(n, ast.Global):
                    for g in n.names:
                        stores[g] = stores.get(g, 0) + 2

        def rewrite(stmts):
            for i, s in enumerate(stmts):
                if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name) \
                        and isinstance(s.value, ast.Lambda) and stores.get(s.targets[0].id) == 1:
                    lam = s.value
                    d = ast.FunctionDef(name=s.targets[0].id, args=copy.deepcopy(lam.args),
                                        body=[_fix(ast.Return(value=lam.body), lam.body)], decorator_list=[], returns=None,
                                        type_comment=None)
                    if hasattr(ast, "TypeAlias"):
                        d.type_params = []
                    stmts[i] = _fix(d, s)
                    stmts[i]._py2viv_lambda = True
                elif fdef is not None:
                    for field in ("body", "orelse", "finalbody"):
                        sub = getattr(s, field, None)
                        if isinstance(sub, list) and not isinstance(s, (ast.FunctionDef, ast.ClassDef)):
                            rewrite(sub)
        rewrite(body)

    # ------------------------------------------------ nested defs
    def lift_nested(self, F, parent_list):
        """Lift the defs nested in F (in any block of F, one level of def) to module level, just
        before F (see the module docstring)."""
        nested = []   # (def node, the statement list holding it)

        def collect(stmts):
            for s in stmts:
                if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    nested.append((s, stmts))
                    continue
                if isinstance(s, ast.ClassDef):
                    continue
                for field in ("body", "orelse", "finalbody"):
                    sub = getattr(s, field, None)
                    if isinstance(sub, list):
                        collect(sub)
                for h in getattr(s, "handlers", []) or []:
                    collect(h.body)
        collect(F.body)
        if not nested:
            return
        fparams = [a.arg for a in F.args.posonlyargs + F.args.args + F.args.kwonlyargs]
        flocals = bound_names(F.body, fparams)
        names = [d.name for d, _ in nested]
        holder_of = {id(d): h for d, h in nested}
        par = _parents(F)
        # names a lambda or a comprehension inside F binds: a nested def's name there would be
        # another variable
        inner_bound = set()
        for n in ast.walk(F):
            if isinstance(n, ast.Lambda):
                inner_bound |= {a.arg for a in n.args.args + n.args.posonlyargs + n.args.kwonlyargs}
            if isinstance(n, ast.comprehension):
                inner_bound |= set(target_names(n.target))
        ok = {}
        for d, holder in nested:
            why = self._nested_refusal(d, F, names)
            if why is None and names.count(d.name) > 1:
                why = ("`%s` defined twice inside `%s`" % (d.name, F.name), "Give each nested def its own name.")
            if why is None and d.name in inner_bound:
                why = ("`%s` both a nested def and a lambda parameter or comprehension variable" % d.name, "Use two names.")
            if why is None:
                for n in walk_scope(F.body):
                    if isinstance(n, ast.Name) and n.id == d.name and isinstance(n.ctx, (ast.Store, ast.Del)):
                        why = ("`%s` both a nested def and a variable of `%s`" % (d.name, F.name), "Use two names.")
                        break
            if why is not None:
                mark(d, *why)
            else:
                ok[d.name] = d
        if not ok:
            return
        # each nested def's own variables, the enclosing variables it reads, the siblings it reads
        dlocals, caps, refs = {}, {}, {}
        for nm, d in ok.items():
            ps = [a.arg for a in d.args.posonlyargs + d.args.args + d.args.kwonlyargs]
            dlocals[nm] = bound_names(d.body, ps)
            fr = set()
            for s in d.body:
                fr |= free_names(s, dlocals[nm])
            caps[nm] = {x for x in fr if x in flocals and x not in ok}
            refs[nm] = {x for x in fr if x in ok}
        # a nested def that reads a sibling passes the sibling's variables on: it reads them too
        changed = True
        while changed:
            changed = False
            for nm in ok:
                for r in refs[nm]:
                    extra = caps[r] - caps[nm]
                    if extra:
                        caps[nm] |= extra
                        changed = True
        bad = set()
        for nm in ok:
            clash = caps[nm] & dlocals[nm]
            if clash:
                v = sorted(clash)[0]
                mark(ok[nm], "`%s` inside `%s`, which reads `%s` of `%s` for a nested def it calls and has a `%s` of its own"
                     % (nm, F.name, v, F.name, v),
                     "Rename its own `%s`: the nested defs it calls read `%s`'s." % (v, F.name))
                bad.add(nm)

        def reach(nm):
            out, stack = set(), [nm]
            while stack:
                x = stack.pop()
                if x not in out:
                    out.add(x)
                    stack.extend(refs[x])
            return out
        for nm in ok:
            if reach(nm) & bad:
                bad.add(nm)
        # every read of a nested def's name, in F or in a nested def
        uses = []  # (name, Name node, owner nested def name or None, the callee of a call)

        def visit(n, owner):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in ok and ok[n.name] is n:
                for s in n.body:
                    visit(s, n.name)
                return
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in ok:
                if owner is None or n.id not in dlocals[owner]:
                    p = par.get(n)
                    uses.append((n.id, n, owner, isinstance(p, ast.Call) and p.func is n))
            for c in ast.iter_child_nodes(n):
                visit(c, owner)
        for s in F.body:
            visit(s, None)
        # Python defines a nested def when its `def` runs, and reads the enclosing variables when
        # it runs: a read of it from F must follow its def (and the defs it reads) in their
        # block, and the assignment of every enclosing variable they read
        stores = {}
        for n in walk_scope(F.body):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                stores.setdefault(n.id, []).append(_pos(n))
            if isinstance(n, ast.For):
                for t in target_names(n.target):
                    stores.setdefault(t, []).append(_pos(n))

        def anchor_refusal(nm, node):
            for r in sorted(reach(nm)):
                d = ok[r]
                holder = holder_of[id(d)]
                x = node
                while x is not None and not any(x is h for h in holder):
                    x = par.get(x)
                if x is None or holder.index(x) <= holder.index(d):
                    return ("`%s` %sread before its `def` runs, or outside the block that defines it"
                            % (r, "" if r == nm else "(which `%s` calls) " % nm),
                            "Define the nested def first, in `%s`'s body, and call it after." % F.name)
                for c in sorted(caps[r]):
                    if c not in fparams and not any(p < _pos(node) for p in stores.get(c, [])):
                        return ("`%s` reads `%s`, which is not assigned before this call" % (r, c),
                                "Python raises NameError; assign `%s` before calling `%s`." % (c, nm))
            return None

        for nm, n, owner, is_call in uses:
            if nm in bad:
                continue
            if owner is None:
                why = anchor_refusal(nm, n)
                if why is not None:
                    mark(par.get(n) if is_call else n, *why)
                    continue
            if not is_call and caps[nm] and not getattr(n, REFUSE, None):
                v = sorted(caps[nm])[0]
                mark(n, "the nested def `%s`, which reads `%s` of `%s`, used as a value" % (nm, v, F.name),
                     "Viv's compiled engines have no closures: call it here, or make it a module-level def that takes `%s`." % v)
        lifted = {nm: self.fresh_def("%s__%s" % (F.name, nm)) for nm in ok if nm not in bad}
        for nm, n, owner, is_call in uses:
            if nm in bad or getattr(n, REFUSE, None) or is_call and getattr(par.get(n), REFUSE, None):
                continue
            if is_call:
                p = par.get(n)
                p.args = [_fix(ast.Name(id=c, ctx=ast.Load()), n) for c in sorted(caps[nm])] + p.args
            n.id = lifted[nm]
        # the lifted defs, before F
        out = []
        for nm, d in ok.items():
            if nm in bad:
                continue
            holder = holder_of[id(d)]
            holder[holder.index(d)] = _fix(ast.Pass(), d)
            args = copy.copy(d.args)
            args.args = [_fix(ast.arg(arg=c, annotation=None), d) for c in sorted(caps[nm])] + list(d.args.args)
            new = ast.FunctionDef(name=lifted[nm], args=args, body=d.body, decorator_list=d.decorator_list,
                                  returns=d.returns, type_comment=None)  # (an int or Optional result: ints.py)
            if hasattr(ast, "TypeAlias"):
                new.type_params = []
            new = _fix(new, d)
            new._py2viv_nested = (F.name, nm, sorted(caps[nm]))
            out.append(new)
        k = next(i for i, s in enumerate(parent_list) if s is F)
        parent_list[k:k] = out

    def fresh_def(self, stem):
        cand = stem
        k = 2
        while cand in self.names:
            cand = "%s_%d" % (stem, k)
            k += 1
        self.names.add(cand)
        return cand

    def _nested_refusal(self, d, F, names):
        if isinstance(d, ast.AsyncFunctionDef):
            return ("`async def`", "Write a plain `def`.")
        if d.decorator_list:
            return ("a decorator", "Drop the decorator.")
        for n in ast.walk(d):
            if n is not d and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return ("a function nested two levels deep (`%s` inside `%s`)" % (getattr(n, "name", "?"), d.name),
                        "Move it up a level.")
            if isinstance(n, ast.Nonlocal):
                return ("a nested def that rebinds `%s` of `%s` (`nonlocal`)" % (n.names[0], F.name),
                        "Return the new value and assign it in `%s`, or keep it in a one-element array." % F.name)
            if isinstance(n, ast.Global):
                return ("`global` inside a nested def", "Pass the value in and return it.")
        for x in d.args.defaults:
            if not is_const_default(x):
                return ("a default computed when the nested def `%s` is defined" % d.name,
                        "Python evaluates it once, when the def runs; pass the value at each call.")
        return None

    # ------------------------------------------------ lambdas passed to defs
    def lambda_args(self):
        """A lambda passed to a def of the program -> a module-level def passed by name, when it
        reads no variable of an enclosing def or comprehension."""
        defs = {s.name for body in self.module_lists() for s in body if isinstance(s, ast.FunctionDef)}
        par = _parents(self.tree)
        for body in self.module_lists():
            i = 0
            while i < len(body):
                top = body[i]
                new_defs = []
                scope = top if isinstance(top, ast.FunctionDef) else None
                local = set()
                if scope is not None:
                    ps = [a.arg for a in scope.args.posonlyargs + scope.args.args + scope.args.kwonlyargs]
                    local = bound_names(scope.body, ps)
                for n in list(ast.walk(top)):
                    if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in defs):
                        continue
                    for k, a in list(enumerate(n.args)) + [(kw, kw.value) for kw in n.keywords]:
                        if not isinstance(a, ast.Lambda):
                            continue
                        fr = free_names(a)
                        comp = _comprehension_names(a, par)
                        bad = sorted((fr & local) | (fr & comp))
                        if bad:
                            mark(a, "a lambda that reads `%s`, a variable of %s, passed to `%s`" % (
                                bad[0], "`%s`" % scope.name if bad[0] in local else "a comprehension", n.func.id),
                                 "Viv's compiled engines have no closures: pass `%s` to `%s` and on to the function." % (bad[0], n.func.id))
                            continue
                        if not all(is_const_default(x) for x in a.args.defaults):
                            mark(a, "a lambda default computed when the lambda is made",
                                 "Python evaluates it once; pass the value at each call.")
                            continue
                        name = self.fresh_def("_lam%d" % a.lineno)
                        d = ast.FunctionDef(name=name, args=copy.deepcopy(a.args),
                                            body=[_fix(ast.Return(value=a.body), a.body)], decorator_list=[], returns=None,
                                            type_comment=None)
                        if hasattr(ast, "TypeAlias"):
                            d.type_params = []
                        new_defs.append(_fix(d, a))
                        ref = _fix(ast.Name(id=name, ctx=ast.Load()), a)
                        if isinstance(k, ast.keyword):
                            k.value = ref
                        else:
                            n.args[k] = ref
                        defs.add(name)
                body[i:i] = new_defs
                i += len(new_defs) + 1


def _comprehension_names(node, par):
    """The names bound by the comprehensions around node."""
    out = set()
    n = par.get(node)
    while n is not None:
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            for g in n.generators:
                out |= set(target_names(g.target))
        n = par.get(n)
    return out


def _is_main_guard(node):
    t = node.test
    return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "__name__"
            and len(t.comparators) == 1 and isinstance(t.comparators[0], ast.Constant)
            and t.comparators[0].value == "__main__" and not node.orelse)


def _fix(new, old):
    ast.copy_location(new, old)
    ast.fix_missing_locations(new)
    for n in ast.walk(new):
        if getattr(n, "lineno", None) is None and hasattr(n, "lineno"):
            ast.copy_location(n, old)
    return new


def lift(tree):
    return Lifter(tree).run()


# ================================================================ the translator's methods
#
# P is py2viv's module (its types and helpers), set by py2viv.py when it loads this one.
P = None


def _host_exec(src):
    """The last line of the traceback this Python prints for the statements `src`."""
    try:
        exec(src, {})
    except Exception as e:  # noqa: BLE001
        m = str(e)
        return "%s: %s" % (type(e).__name__, m) if m else type(e).__name__
    return None


HOST = {
    "max_empty": _host_exec("max([])"),
    "min_empty": _host_exec("min([])"),
    "pop_empty": _host_exec("[].pop()"),
    "pop_index": _host_exec("[1].pop(5)"),
    "remove_missing": _host_exec("[1].remove(2)"),
    "range_zero": _host_exec("range(1, 2, 0)"),
    "unpack_many": _host_exec("a, b = [1, 2, 3]"),
    "unpack_many_4": _host_exec("a, b = [1, 2, 3, 4]"),
    "stop_iteration": "StopIteration",
}


def _is_call(n, name):
    return isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name


def join_all(ts):
    """The type of values of these types in one place (None when they cannot share it)."""
    out = None
    for t in ts:
        out = P.join(out, t)
        if out is None:
            return None
    return out


class FuncsMixin:
    """The translator's methods for Python functions and lists (see the module docstring)."""

    # ------------------------------------------------ comprehension scopes
    # A comprehension's targets, a lambda's parameters, and a generator's variables are local to
    # them in Python: each is bound to a fresh Viv name in a scope of its own, so it never
    # overwrites a variable of the same name outside.

    def _comp_get(self, name):
        for env in reversed(getattr(self, "comp_env", None) or ()):
            if name in env:
                return env[name]
        return None

    def _comp_push(self):
        if getattr(self, "comp_env", None) is None:
            self.comp_env = []
        env = {}
        self.comp_env.append(env)
        return env

    def _comp_pop(self):
        self.comp_env.pop()

    def _comp_binder(self, env):
        def bind(name, code, t, lines, ind, direct=False):
            if direct:
                env[name] = (code, t)
                return
            tmp = self.new_tmp(name)
            lines.append(ind + "%s = %s" % (tmp, code))
            env[name] = (tmp, t)
        return bind

    def _stmt_binder(self, s):
        """Bind a for statement's own targets: ordinary variables (Python's for loop leaks them)."""
        def bind(name, code, t, lines, ind, direct=False):
            self.set_var(name, t, s)
            lines.append(ind + "%s = %s" % (self.vname(name), code))
        return bind

    def _sub(self, node, lines, ind):
        """Translate an expression whose hoisted statements go into lines at ind."""
        saved = self.pre
        self.pre = []
        try:
            r = self.expr(node)
        finally:
            got = self.pre
            self.pre = saved
        lines.extend(ind + x for x in got)
        return r

    def _sub_cond(self, node, lines, ind):
        saved = self.pre
        self.pre = []
        try:
            c = self._cond(node)
        finally:
            got = self.pre
            self.pre = saved
        lines.extend(ind + x for x in got)
        return c

    # ------------------------------------------------ the clauses of a comprehension
    def _open_gens(self, gens, node, bind, lines, ind, stack):
        """Open the loops and `if`s of a comprehension's clauses into lines, binding its targets
        with bind; stack records what was opened ("loop" or "if"). Returns the inside's
        indentation."""
        for g in gens:
            if g.is_async:
                self.refuse(node, "an `async for` clause", "Write a plain comprehension.")
            ind = self._open_clause(g.iter, g.target, node, bind, lines, ind, stack)
            for c in g.ifs:
                cc = self._sub_cond(c, lines, ind)
                lines.append(ind + "if %s {" % cc)
                stack.append("if")
                ind += "    "
        return ind

    @staticmethod
    def _close(lines, ind, stack, brk=None):
        """Close what _open_gens opened; after an inner loop, `if <brk> { break }` leaves the
        outer ones too (a Viv break leaves one loop)."""
        while stack:
            kind = stack.pop()
            ind = ind[:-4]
            lines.append(ind + "}")
            if kind == "loop" and brk is not None and "loop" in stack:
                lines.append(ind + "if %s {" % brk)
                lines.append(ind + "    break")
                lines.append(ind + "}")
        return ind

    def _range_parts(self, call, lines, ind):
        """range(...)'s arguments: (start, stop, step) codes, ints (TypeError otherwise), a zero
        step refused or stopped as Python raises."""
        if call.keywords or not 1 <= len(call.args) <= 3 or any(isinstance(a, ast.Starred) for a in call.args):
            self.refuse(call, "range with these arguments", "Pass one to three ints.")
        parts = []
        for a in call.args:
            c, t, p = self._sub(a, lines, ind)
            if t.kind == "bool" or t.kind not in ("num", "unknown") or t.py in ("float", "mixed"):
                self.refuse(a, "range of %s" % ("a float" if t.kind == "num" else self._kind_words(t)),
                            "Python raises TypeError: range takes ints.")
            parts.append((c, p))
        if len(parts) == 1:
            return "0", parts[0][0], "1"
        start, stop = parts[0][0], parts[1][0]
        if len(parts) == 2:
            return start, stop, "1"
        step, sp = parts[2]
        st = call.args[2]
        lit = st.value if isinstance(st, ast.Constant) else (-st.operand.value if isinstance(st, ast.UnaryOp) and isinstance(
            st.op, ast.USub) and isinstance(st.operand, ast.Constant) else None)
        if lit == 0:
            self.refuse(st, "a range step of 0", "Python raises ValueError; use a nonzero step.")
        if lit is None:
            if not P.re.fullmatch(r"[A-Za-z_]\w*", step):
                tmp = self.new_tmp("s")
                lines.append(ind + "%s = %s" % (tmp, step))
                step = tmp
            lines.append(ind + "if %s == 0 {" % step)
            lines.append(ind + "    " + self._host_stop("range_zero", call))
            lines.append(ind + "}")
        return start, stop, step

    def _host_stop(self, key, node, exact=True):
        msg = HOST[key]
        self.stops.append({"line": getattr(node, "lineno", 0), "message": msg, "exact": exact})
        return "stop(%s)" % P.viv_str(msg)

    @staticmethod
    def _range_count(start, stop, step):
        if step == "1":
            return stop if start == "0" else "max(%s - %s, 0)" % (stop, start)
        return "max(ceil((%s - %s) / %s), 0)" % (stop, start, step)

    def _open_clause(self, it, tgt, node, bind, lines, ind, stack):
        """One `for tgt in it` clause: the loop head into lines; returns the inside's indentation."""
        rev = False
        if _is_call(it, "reversed") and len(it.args) == 1 and not it.keywords and _is_call(it.args[0], "range"):
            rev, it = True, it.args[0]
        if _is_call(it, "range") and isinstance(tgt, ast.Name):
            start, stop, step = self._range_parts(it, lines, ind)
            v = self.new_tmp(tgt.id)
            if start == "0" and step == "1":
                head = "for %s to %s" % (v, stop)
            elif step == "1":
                head = "for %s from %s to %s" % (v, start, stop)
            else:
                head = "for %s from %s to %s by %s" % (v, start, stop, step)
            lines.append(ind + head + (" reversed" if rev else "") + " {")
            stack.append("loop")
            ind += "    "
            bind(tgt.id, v, P.NUM_I, lines, ind, direct=True)
            return ind
        if rev:
            it = ast.copy_location(ast.Call(func=ast.Name(id="reversed", ctx=ast.Load()), args=[it], keywords=[]), it)
        if isinstance(it, ast.GeneratorExp):
            # a generator's items (map and filter become one): its clauses, then the target
            # is its element
            env = self._comp_push()
            try:
                ind = self._open_gens(it.generators, it, self._comp_binder(env), lines, ind, stack)
                code, t, _ = self._sub(it.elt, lines, ind)
            finally:
                self._comp_pop()
            self._bind_target(tgt, code, t, bind, lines, ind, node)
            return ind
        # indexed sources: enumerate(seq, start), zip(a, b, ...), a sequence
        if _is_call(it, "enumerate") and self._no_def("enumerate"):
            if not (isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2):
                self.refuse(tgt, "this enumerate target", "Write `for i, x in enumerate(a)`.")
            if len(it.args) not in (1, 2) or any(k.arg != "start" for k in it.keywords) or len(it.args) + len(it.keywords) > 2:
                self.refuse(it, "enumerate with these arguments", "Pass the sequence and, at most, the start.")
            start = "0"
            snode = it.args[1] if len(it.args) == 2 else (it.keywords[0].value if it.keywords else None)
            srcs = [self._source(it.args[0], lines, ind, node)]
            if snode is not None:
                c, t, p = self._sub(snode, lines, ind)
                if t.kind != "num" or t.py != "int":
                    self.refuse(snode, "an enumerate start that is not an int", "Python raises TypeError; pass an int.")
                if not P.re.fullmatch(r"[A-Za-z_]\w*|-?\d+", c):
                    tmp = self.new_tmp("e")
                    lines.append(ind + "%s = %s" % (tmp, c))
                    c = tmp
                start = c
            srcs.insert(0, ("index", start))
            tgts = tgt.elts
        elif _is_call(it, "zip") and self._no_def("zip"):
            if it.keywords:
                self.refuse(it, "zip with keywords (strict=)", "Pass the sequences only.")
            if not it.args:
                self.refuse(it, "zip of nothing", "Pass the sequences.")
            if not (isinstance(tgt, ast.Tuple) and len(tgt.elts) == len(it.args)):
                self.refuse(tgt, "this zip target", "Unpack one name per zipped sequence.")
            srcs = [self._source(a, lines, ind, node) for a in it.args]
            tgts = tgt.elts
        else:
            srcs = [self._source(it, lines, ind, node)]
            tgts = [tgt]
        counts = [s[-1] for s in srcs if s[0] != "index"]
        count = counts[0]
        for c in counts[1:]:
            count = "min(%s, %s)" % (count, c)
        k = self.new_tmp("i")
        lines.append(ind + "for %s to %s {" % (k, count))
        stack.append("loop")
        ind += "    "
        for src, tg in zip(srcs, tgts):
            code, t = self._source_item(src, k, node, tg)
            self._bind_target(tg, code, t, bind, lines, ind, node)
        return ind

    def _no_def(self, name):
        return name not in self.funcs and self.var_type(name) is None

    def _source(self, e, lines, ind, node):
        """A sequence read by index: ("range", start, step, count) or ("seq", code, type,
        reversed, node, count)."""
        rev = False
        if _is_call(e, "reversed") and len(e.args) == 1 and not e.keywords and self._no_def("reversed"):
            rev, e = True, e.args[0]
        if _is_call(e, "range") and self._no_def("range"):
            start, stop, step = self._range_parts(e, lines, ind)
            cnt = self._range_count(start, stop, step)
            if rev:
                tmp = self.new_tmp("n")
                lines.append(ind + "%s = %s" % (tmp, cnt))
                last = "%s + (%s - 1) * %s" % (start, tmp, step) if step != "1" else "%s + %s - 1" % (start, tmp)
                if start == "0" and step == "1":
                    last = "%s - 1" % tmp
                s2 = self.new_tmp("r")
                lines.append(ind + "%s = %s" % (s2, last))
                neg = "-1" if step == "1" else "0 - %s" % step
                return ("range", s2, neg, tmp)
            return ("range", start, step, cnt)
        if isinstance(e, ast.GeneratorExp):
            self.refuse(e, "a generator inside enumerate, zip or reversed", "Build the list first: [... for ...].")
        fl = self._lines_of_file(e)  # open(p), a `with` file: its lines (strs.py)
        if fl is not None:
            code, t = fl, P.STRS
        else:
            code, t, _ = self._sub(e, lines, ind)
        if t.kind in ("str", "strs"):
            # a string's characters, a list's strings (strs.py)
            if not P.re.fullmatch(r"[A-Za-z_]\w*", code):
                tmp = self.new_tmp("s")
                lines.append(ind + "%s = %s" % (tmp, code))
                code = tmp
            return ("seq", code, t, rev, e, "len(%s)" % code)
        if t.kind == "unknown":
            r = P.Refusal(e, "iterating over a value of unknown type", "Fix the refusal that produced it.")
            r.cascade = True
            raise r
        if not t.is_arr() or t.container == "shape":
            self.refuse(e, "iterating over %s" % self._kind_words(t), "Loop over `range(len(a))` and index.")
        if t.rank > 2:
            self.refuse(e, "iterating over %s" % self._kind_words(t), "Loop over the indices of each axis.")
        if not P.re.fullmatch(r"[A-Za-z_]\w*", code):
            tmp = self.new_tmp("s")
            lines.append(ind + "%s = %s" % (tmp, code))
            code = tmp
        return ("seq", code, t, rev, e, "len(%s)" % code)

    def _source_item(self, src, k, node, tgt):
        if src[0] == "index":
            return (k if src[1] == "0" else "%s + %s" % (src[1], k)), P.NUM_I
        if src[0] == "range":
            _, start, step, _ = src
            if start == "0" and step == "1":
                return k, P.NUM_I
            if step == "1":
                return "%s + %s" % (start, k), P.NUM_I
            return "%s + %s * %s" % (start, k, self._paren_step(step)), P.NUM_I
        _, code, t, rev, e, _ = src
        idx = "len(%s) - 1 - %s" % (code, k) if rev else k
        if t.kind in ("str", "strs"):
            return "%s[%s]" % (code, idx), P.STR
        if t.rank == 1:
            return self._elem_of(code, idx, t)
        # a row of a matrix (a view in NumPy, a row object shared by a list of lists)
        names = [x.id for x in ast.walk(tgt) if isinstance(x, ast.Name)]
        self._row_alias_check(e, node, names=names)
        return "slab(%s, %s)" % (code, idx), P.row_type(t)

    @staticmethod
    def _paren_step(step):
        return step if P.re.fullmatch(r"[A-Za-z_]\w*|\d+", step) else "(%s)" % step

    @staticmethod
    def _elem_of(code, idx, t):
        """An entry of a vector, typed as Python sees it."""
        if t.kind == "carr":
            return "%s[%s]" % (code, idx), P.CNUM
        if t.py == "bool":
            if t.container == "list":
                return "%s[%s] != 0" % (code, idx), P.BOOL
            return "%s[%s]" % (code, idx), P.NUM_I
        if t.elts:
            et = join_all(t.elts) or P.NUM_MIXED
            if et.kind == "bool":
                return "%s[%s] != 0" % (code, idx), P.BOOL
            return "%s[%s]" % (code, idx), et
        return "%s[%s]" % (code, idx), {"int": P.NUM_I, "float": P.NUM_F}.get(t.py, P.NUM_MIXED)

    def _bind_target(self, tgt, code, t, bind, lines, ind, node):
        """Bind a clause's target to an item: a name, or a tuple of names unpacking a row."""
        if isinstance(tgt, ast.Name):
            if t.kind == "bool" and not P.re.fullmatch(r"[A-Za-z_]\w*|true|false", code):
                code = "(%s)" % code
            bind(tgt.id, code, t, lines, ind)
            return
        if isinstance(tgt, (ast.Tuple, ast.List)):
            if not (t.kind in ("arr", "carr") and t.rank == 1):
                self.refuse(tgt, "unpacking %s" % self._kind_words(t), "Unpack a row of a matrix or a list of rows.")
            n = len(tgt.elts)
            if any(isinstance(x, ast.Starred) for x in tgt.elts):
                self.refuse(tgt, "a starred target", "Unpack into as many names as there are values.")
            row = code
            if not P.re.fullmatch(r"[A-Za-z_]\w*", row):
                row = self.new_tmp("w")
                lines.append(ind + "%s = %s" % (row, code))
            self._unpack_check(row, n, lines, ind, tgt)
            for j, x in enumerate(tgt.elts):
                et = t.elts[j] if t.elts and j < len(t.elts) else None
                c, tt = self._elem_of(row, str(j), t)
                if et is not None and et.kind == "bool" and tt.kind != "bool":
                    c = "%s != 0" % c  # a Bool item is stored as 1 or 0
                self._bind_target(x, c, et or tt, bind, lines, ind, node)
            return
        self.refuse(tgt, "this loop target", "Loop over a name, or a tuple of names.")

    def _unpack_check(self, row, n, lines, ind, node):
        """Python's unpacking of a row of the wrong length raises ValueError: the same stop, with
        this CPython's words (a wording that names the length found is compared by its type)."""
        names = ", ".join("a%d" % i for i in range(n)) + ("," if n == 1 else "")
        many = _host_exec("%s = [0] * %d" % (names, n + 1))
        exact = many == _host_exec("%s = [0] * %d" % (names, n + 2))
        self.stops.append({"line": node.lineno, "message": many, "exact": exact})
        lines.append(ind + "if len(%s) > %d {" % (row, n))
        lines.append(ind + "    stop(%s)" % P.viv_str(many))
        lines.append(ind + "}")
        for got in range(n):
            msg = _host_exec("%s = [0] * %d" % (names, got))
            self.stops.append({"line": node.lineno, "message": msg, "exact": True})
            lines.append(ind + "if len(%s) == %d {" % (row, got))
            lines.append(ind + "    stop(%s)" % P.viv_str(msg))
            lines.append(ind + "}")


    # ------------------------------------------------ list comprehensions
    def _e_ListComp(self, n):
        """[e for ... in ... if ...]: a new list. One clause over a sequence or a range, no
        condition, a number element: allocated at its length and filled. Anything else: grown
        by Viv's append, rows of a matrix when the element is a
        list. Its clauses are loops, its conditions ifs, its variables its own."""
        g = n.generators
        R = self.new_tmp("l")
        lines, stack = [], []
        env = self._comp_push()
        try:
            ind = self._open_gens(g, n, self._comp_binder(env), lines, "", stack)
            code, t, _ = self._sub(n.elt, lines, ind)
            sized = self._sized_loop(g, lines, stack)
            if sized is not None and t.kind in ("num", "bool") and not isinstance(n.elt, ast.Tuple):
                # one clause over a range or a sequence, no condition, a number: its length is
                # known, so the list is allocated and filled
                count, idx = sized
                rt = self._list_add(R, code, t, n.elt, [], ind, n)
                if t.kind == "bool":
                    code = {"true": "1", "false": "0"}.get(code) or "%s(%s)" % (self.use("py_b01"), code)
                lines.append(ind + "%s[%s] = %s" % (R, idx, code))
                self._close(lines, ind, stack)
                comp = self._viv_comprehension(lines, R, idx)
                if comp is not None:
                    return comp, rt, 100
                head = next(k for k, x in enumerate(lines) if x.startswith("for "))
                lines.insert(head, "%s = zeros(%s)" % (R, count))  # after the sequence's own statements
                self.pre.extend(lines)
                return R, rt, 100
            rt = self._list_add(R, code, t, n.elt, lines, ind, n)
        finally:
            self._comp_pop()
        self._close(lines, ind, stack)
        comp = self._viv_comprehension(lines, R, None)
        if comp is not None:
            return comp, rt, 100
        self.pre.append("%s = []" % R)
        self.pre.extend(lines)
        return R, rt, 100

    @staticmethod
    def _viv_comprehension(lines, R, idx):
        """Viv's comprehension for the loops that build R, when they are only loop heads, `if`s
        and the one statement that adds the item (`R = append(R, e)`, or `R[idx] = e` when the
        list is allocated at its length); None otherwise. Viv writes its comprehension as the
        same loops (lang/viv/src/parser.rs), so the value is the same. A head `for k to len(s)`
        whose counter only reads `x = s[k]` is `for x in s`."""
        body = [x.strip() for x in lines]
        clauses, k, opened = [], 0, 0
        while k < len(body):
            m = P.re.fullmatch(r"for (\w+) ((?:to|from) .+) \{", body[k])
            if m:
                v, rest = m.group(1), m.group(2)
                seq = P.re.fullmatch(r"to len\((\w+)\)", rest)
                nxt = P.re.fullmatch(r"(\w+) = %s\[%s\]" % (P.re.escape(seq.group(1)), P.re.escape(v)), body[k + 1]) \
                    if seq and k + 1 < len(body) else None
                others = [x for j, x in enumerate(body) if j not in (k, k + 1)]
                if idx == v:
                    others = [x for x in others if not x.startswith("%s[%s] = " % (R, v))]
                two = P.re.fullmatch(r"to min\(len\((\w+)\), len\((\w+)\)\)", rest)
                pair = [P.re.fullmatch(r"(\w+) = %s\[%s\]" % (P.re.escape(two.group(j)), P.re.escape(v)), body[k + j])
                        if k + j < len(body) else None for j in (1, 2)] if two else [None, None]
                rest_zip = [x for j, x in enumerate(body) if j not in (k, k + 1, k + 2)]
                if idx == v:
                    rest_zip = [x for x in rest_zip if not x.startswith("%s[%s] = " % (R, v))]
                if nxt and not any(P.re.search(r"\b%s\b" % P.re.escape(v), x) for x in others):
                    clauses.append("for %s in %s" % (nxt.group(1), seq.group(1)))
                    k += 2
                elif all(pair) and not any(P.re.search(r"\b%s\b" % P.re.escape(v), x) for x in rest_zip):
                    clauses.append("for %s, %s in zip(%s, %s)" % (pair[0].group(1), pair[1].group(1), two.group(1), two.group(2)))
                    k += 3
                else:
                    if idx is not None:
                        return None
                    clauses.append("for %s %s" % (v, rest))
                    k += 1
                opened += 1
                continue
            m = P.re.fullmatch(r"if (.+) \{", body[k])
            if m:
                clauses.append("if %s" % m.group(1))
                opened += 1
                k += 1
                continue
            break
        if not clauses or not clauses[0].startswith("for ") or k >= len(body):
            return None
        if idx is None:
            m = P.re.fullmatch(r"%s = append\(%s, (.+)\)" % (P.re.escape(R), P.re.escape(R)), body[k])
        else:
            m = P.re.fullmatch(r"%s\[%s\] = (.+)" % (P.re.escape(R), P.re.escape(idx)), body[k])
        if m is None or body[k + 1:] != ["}"] * opened:
            return None
        e = m.group(1)
        if idx is not None and P.re.search(r"\b%s\b" % P.re.escape(idx), e):
            return None
        if R in e or any(R in c for c in clauses):
            return None
        return "[%s %s]" % (e, " ".join(clauses))

    @staticmethod
    def _sized_loop(g, lines, stack):
        """(count, index) of a comprehension of one clause and no condition whose loop runs
        over `for k to count` (a range from 0 by 1, a sequence, enumerate, zip); else None."""
        if len(g) != 1 or g[0].ifs or stack != ["loop"]:
            return None
        heads = [x for x in lines if x.startswith("for ")]
        if len(heads) != 1:
            return None
        m = P.re.fullmatch(r"for (\w+) to (.+) \{", heads[0])
        if m is None:
            return None
        return m.group(2), m.group(1)

    def _list_add(self, R, code, t, elt, lines, ind, node):
        """`R = append(R, e)` for one item of a new list; the list's type."""
        if isinstance(elt, ast.Tuple):
            return self._tuple_row(R, elt, lines, ind, node)
        if t.kind == "unknown":
            r = P.Refusal(elt, "a list of values of unknown type", "Fix the refusal that produced it.")
            r.cascade = True
            raise r
        if t.kind in ("num", "bool"):
            if t.py == "mixed":
                self.refuse(elt, "a list whose items are ints on some paths and floats on others",
                            "Python keeps each as it is (and prints 3 or 3.0); convert them with float(...).")
            if t.kind == "bool":
                code = {"true": "1", "false": "0"}.get(code) or "%s(%s)" % (self.use("py_b01"), code)
            lines.append(ind + "%s = append(%s, %s)" % (R, R, code))
            return P.arr(1, "bool" if t.kind == "bool" else t.py, "list")
        if t.kind == "arr" and t.rank == 1:
            if t.py in ("mixed",):
                self.refuse(elt, "a list of rows mixing ints and floats", "Make them all floats.")
            self._fresh_row(elt, node)
            row = code
            if not P.re.fullmatch(r"[A-Za-z_]\w*", row):
                row = self.new_tmp("w")
                lines.append(ind + "%s = %s" % (row, code))
            self._ragged_check(R, row, lines, ind)
            lines.append(ind + "%s = append(%s, %s)" % (R, R, row))
            # rows that are NumPy arrays stay arrays in Python's list (printed `array([...])`)
            return P.T("arr", 2, t.py, "list", elts=t.elts, name=t.name if t.container != "nd" else "ndrows")
        if t.kind == "str":
            lines.append(ind + "%s = append(%s, %s)" % (R, R, code))
            return P.STRS  # a list of strings (strs.py)
        if t.kind in ("cnum", "carr"):
            self.refuse(elt, "a list of complex numbers built item by item", "Viv's append takes real numbers; "
                        "allocate np.zeros(n, dtype=complex) and fill it.")
        self.refuse(elt, "a list of %s" % ("matrices" if t.is_arr() else self._kind_words(t).replace("a ", "", 1) + "s"),
                    "A Viv list holds numbers, or rows of numbers.")

    def _tuple_row(self, R, elt, lines, ind, node):
        """A tuple of numbers as an item: a row whose entries keep their own types (a tuple
        cannot change, so the row is the list's own)."""
        codes, types = [], []
        for e in elt.elts:
            if isinstance(e, ast.Starred):
                self.refuse(e, "a starred item in a tuple", "Write the items one by one.")
            c, t, _ = self._sub(e, lines, ind)
            if t.kind == "bool":
                c = {"true": "1", "false": "0"}.get(c) or "%s(%s)" % (self.use("py_b01"), c)
            elif t.kind != "num" or t.py == "mixed":
                self.refuse(e, "a tuple holding %s in a list" % (
                    "a number that is an int on some paths and a float on others" if t.kind == "num" else self._kind_words(t)),
                    "A list of tuples py2viv builds holds tuples of numbers.")
            codes.append(c)
            types.append(t)
        if not codes:
            self.refuse(elt, "an empty tuple in a list", "Store numbers.")
        row = self.new_tmp("w")
        lines.append(ind + "%s = [%s]" % (row, ", ".join(codes)))
        self._ragged_check(R, row, lines, ind)
        lines.append(ind + "%s = append(%s, %s)" % (R, R, row))
        py = "float" if any(t.py == "float" for t in types) else "int"
        return P.T("arr", 2, py, "list", elts=types, name="tuple")

    def _ragged_check(self, R, row, lines, ind):
        """A Viv matrix has one row length; a Python list of lists may not: stop (a run-time
        refusal) where the rows would differ."""
        lines.append(ind + "if len(%s) > 0 {" % R)
        lines.append(ind + "    if shape(%s)[1] != len(%s) {" % (R, row))
        lines.append(ind + '        stop("py2viv: a list of rows of different lengths; Viv\'s rows share one length")')
        lines.append(ind + "    }")
        lines.append(ind + "}")

    def _fresh_row(self, e, node):
        """A row put into a new list must be a new list or array: a list of lists shares its rows
        with wherever they came from, and Viv's matrix copies them, so a later write into one
        would show in Python only."""
        if isinstance(e, (ast.List, ast.ListComp, ast.Tuple, ast.BinOp, ast.UnaryOp)):
            return
        if isinstance(e, ast.Call):
            name = self._call_name(e.func)
            f = e.func
            if name in ("list", "sorted", "np.array", "np.copy") and e.args and self._rank1(e.args[0]):
                return
            if isinstance(f, ast.Attribute) and f.attr == "copy" and not e.args and self._rank1(f.value):
                return
            if name and name.startswith("np.") and name not in P.views_mod.VIEW_FUNCS and not self.views.roots(e):
                return
        # a row other names can reach: shared in Python and copied in Viv, the same while no row
        # of any list or matrix of the program ever changes
        site = self._row_write()
        if site is None:
            return
        self.refuse(e, "a row that another name can reach, put into a list, in a program that writes into rows "
                    "(line %d)" % getattr(site, "lineno", 0),
                    "A list of rows shares the row objects in Python and Viv's matrix copies them; "
                    "put a copy in: list(r), r.copy(), or a new list.")

    def _row_write(self):
        """The first write anywhere in the program that could change a row of a list or a matrix
        (a store into an array of another rank than 1, or into a name bound to a row), or None."""
        memo = getattr(self, "_row_write_memo", None)
        if memo is not None and memo[0] is self.out:
            return memo[1]
        found = None
        scopes = [None] + list(self.funcs.values())
        for scope in scopes:
            sname = scope.name if scope is not None else "<module>"
            _, body = self.views.scope_body(scope)
            rows = set()
            for n in walk_scope(body):
                if isinstance(n, (ast.For, ast.comprehension)) and not (isinstance(n.iter, ast.Call) and
                                                                         isinstance(n.iter.func, ast.Name) and n.iter.func.id == "range"):
                    r = self._rank_of(sname, n.iter.id) if isinstance(n.iter, ast.Name) else None
                    if r != 1:
                        rows |= set(target_names(n.target))
                if isinstance(n, ast.Assign) and isinstance(n.value, ast.Subscript):
                    for t in n.targets:
                        rows |= set(target_names(t))
            for root, node, how in self.views.write_sites(body):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) \
                        and node.func.value.id == root and node.func.attr in LIST_MUTATORS - {"sort"}:
                    continue  # a list's own method changes which rows it holds, not a row
                r = self._rank_of(sname, root)
                if r != 1 or root in rows:
                    found = node
                    break
            if found is not None:
                break
        self.views.scope_body(self.scope)
        self._row_write_memo = (self.out, found)
        return found

    def _rank1(self, e):
        saved = self.pre
        try:
            _, t, _ = self.expr(e)
        finally:
            self.pre = saved
        return t.is_arr() and t.rank == 1

    # ------------------------------------------------ sum, any, all over a generator
    def _reduce_gen(self, g, fn):
        """sum/all/any of a generator expression: an accumulator over its loops, before the
        statement; any and all stop at the first item that decides them (through every loop)."""
        acc = self.new_tmp(fn)
        lines, stack = [], []
        env = self._comp_push()
        try:
            ind = self._open_gens(g.generators, g, self._comp_binder(env), lines, "", stack)
            ec, et, ep = self._sub(g.elt, lines, ind)
            init, upd, after, rt = self._reduce_item(fn, acc, ec, et, ep, g)
            lines.extend(ind + x for x in upd)
        finally:
            self._comp_pop()
        self._close(lines, ind, stack, brk={"any": acc, "all": "not %s" % acc}.get(fn))
        self.pre.extend(init)
        self.pre.extend(lines)
        self.pre.extend(after)
        return acc, rt, 100

    def _reduce_item(self, fn, acc, ec, et, ep, g):
        if fn == "sum":
            if et.kind not in ("num", "unknown", "bool"):
                self.refuse(g.elt, "summing %s" % self._kind_words(et), "Sum numbers.")
            if et.py == "mixed":
                self.refuse(g.elt, "summing values that are ints on some paths and floats on others", "Make them all floats.")
            if et.kind == "bool":
                ec = "%s(%s)" % (self.use("py_b01"), ec)
            if et.py in ("int", "bool"):
                return ["%s = 0" % acc], ["%s = %s(%s + %s)" % (acc, self.use("py_int"), acc, ec)], [], P.NUM_I
            if P.HOST_PY >= (3, 12):
                if self.np or self.from_np:
                    self.refuse(g, "the builtin sum of floats in a program that uses NumPy, under Python %d.%d" % P.HOST_PY,
                                "From 3.12 the builtin sum compensates exact Python floats and adds NumPy's float64 "
                                "plainly, and py2viv cannot tell which these items are; sum with np.sum, or add in a loop.")
                x, t_, c_ = self.new_tmp("x"), self.new_tmp("t"), acc + "c"
                upd = ["%s = %s" % (x, ec), "%s = %s + %s" % (t_, acc, x),
                       "if abs(%s) >= abs(%s) {" % (acc, x), "    %s += (%s - %s) + %s" % (c_, acc, t_, x), "}",
                       "else {", "    %s += (%s - %s) + %s" % (c_, x, t_, acc), "}", "%s = %s" % (acc, t_)]
                after = ["if %s != 0 and abs(%s) < 1.0 / 0.0 {" % (c_, c_), "    %s += %s" % (acc, c_), "}"]
                return ["%s = 0" % acc, "%s = 0.0" % c_], upd, after, P.NUM_F
            return ["%s = 0" % acc], ["%s += %s" % (acc, ec)], [], P.NUM_F
        c = self._truth(g.elt, ec, et, ep)
        if fn == "all":
            return ["%s = true" % acc], ["if not (%s) {" % c, "    %s = false" % acc, "    break", "}"], [], P.BOOL
        return ["%s = false" % acc], ["if %s {" % c, "    %s = true" % acc, "    break", "}"], [], P.BOOL

    # ------------------------------------------------ calls of lambdas and of function values
    def _call_lambda(self, n):
        """(lambda x, y: e)(a, b) -- how a key function, map and filter reach here: the
        arguments are evaluated in order, then e with the parameters bound to them, in a scope
        of their own. A lambda reads the variables it does not bind when it runs, which is now."""
        lam = n.func
        a = lam.args
        if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs:
            self.refuse(lam, "a lambda with `*args`, `**kwargs` or keyword-only parameters", "List the parameters positionally.")
        params = [x.arg for x in a.args]
        defaults = dict(zip(params[len(params) - len(a.defaults):], a.defaults))
        given = {}
        if any(isinstance(x, ast.Starred) for x in n.args) or any(k.arg is None for k in n.keywords):
            self.refuse(n, "a starred argument", "Pass the values one by one.")
        if len(n.args) > len(params):
            self.refuse(n, "a lambda called with %d arguments for %d parameters" % (len(n.args), len(params)),
                        "Python raises TypeError.")
        for p, x in zip(params, n.args):
            given[p] = x
        for k in n.keywords:
            if k.arg not in params or k.arg in given:
                self.refuse(k.value, "a lambda called with the keyword `%s`" % k.arg, "Python raises TypeError.")
            given[k.arg] = k.value
        bound = {}
        order = list(n.args) + [k.value for k in n.keywords]
        vals = {}
        for x in order:
            c, t, p = self.expr(x)
            if not P.re.fullmatch(r"[A-Za-z_]\w*|\d+(\.\d+)?|true|false", c):
                tmp = self.new_tmp("a")
                self.pre.append("%s = %s" % (tmp, c))
                c = tmp
            vals[id(x)] = (c, t)
        for p in params:
            if p in given:
                bound[p] = vals[id(given[p])]
            elif p in defaults:
                if not is_const_default(defaults[p]):
                    self.refuse(defaults[p], "a lambda default computed when the lambda is made",
                                "Python evaluates it once; pass the value.")
                c, t, _ = self.expr(defaults[p])
                bound[p] = (c, t)
            else:
                self.refuse(n, "a lambda called without `%s`" % p, "Python raises TypeError.")
        env = self._comp_push()
        env.update(bound)
        try:
            return self.expr(lam.body)
        finally:
            self._comp_pop()

    def _apply_fn(self, fnode, args, node):
        """fnode (a lambda, a def, a builtin) applied to values already in Viv names:
        args = [(code, type)]. Returns (code, type, prec)."""
        env = self._comp_push()
        names = []
        try:
            for k, (c, t) in enumerate(args):
                nm = "_py2viv_arg%d" % k
                env[nm] = (c, t)
                names.append(ast.copy_location(ast.Name(id=nm, ctx=ast.Load()), node))
            call = ast.copy_location(ast.Call(func=fnode, args=names, keywords=[]), node)
            ast.fix_missing_locations(call)
            return self.expr(call)
        finally:
            self._comp_pop()

    # a parameter a def of the program is passed as a value: calls through it
    def _fn_param_index(self, name):
        f = self.scope
        if f is None or name not in f.params or self._comp_get(name) is not None:
            return None
        t = self.var_type(name)
        if t is None or t.kind != "func" or t.name is not None:
            return None
        return f.params.index(name)

    def _flows(self):
        if getattr(self, "fn_flows", None) is None:
            self.fn_flows = {}
        return self.fn_flows

    def _fn_arg(self, f, i, p, t, argnode):
        """A function passed to parameter p (index i) of the def f: a def of the program, or a
        function parameter of the caller passed on. The callee's parameter is typed a function,
        and the defs that reach it are recorded (its calls are typed by them)."""
        flows = self._flows()
        key = (f.name, i)
        if t.name is not None and t.name in self.funcs:
            if self.funcs[t.name].is_test:
                self.refuse(argnode, "passing a test function", "Pass a helper def.")
            flows.setdefault(key, set()).add(t.name)
        elif t.name is None and isinstance(argnode, ast.Name) and self._fn_param_index(argnode.id) is not None:
            src = (self.scope.name, self._fn_param_index(argnode.id))
            flows.setdefault(key, set()).update(flows.get(src, set()))
            flows.setdefault(("fwd",) + src, set()).add(key)
        else:
            self.refuse(argnode, "passing a builtin or library function (`%s`)" % (t.name or "?"),
                        "Viv passes defs of the program: wrap it in a def.")
        old = f.param_types[i]
        if old is not None and old.kind not in ("func", "unknown"):
            self.refuse(argnode, "`%s` called with a function for `%s`, and elsewhere with %s" % (f.name, p, self._kind_words(old)),
                        "Write two functions, one per kind of argument.")
        f.param_types[i] = P.T("func")

    def _call_fn_param(self, n):
        """f(x) where f is a parameter the callers pass defs to: Viv calls through it (its
        compiled engines specialize the def per function passed, design: lang/viv/src/defun.rs).
        Each def that reaches it is typed by this call."""
        name = n.func.id
        i = self._fn_param_index(name)
        f = self.scope
        if n.keywords or any(isinstance(a, ast.Starred) for a in n.args):
            self.refuse(n, "keyword or starred arguments in a call through the parameter `%s`" % name,
                        "Pass the arguments positionally.")
        cands = sorted(self._flows().get((f.name, i), set()))
        codes, types = [], []
        for a in n.args:
            c, t, _ = self.expr(a)
            if t.kind in ("str", "none", "tuple", "module", "func"):
                self.refuse(a, "passing %s through a function parameter" % self._kind_words(t), "Pass numbers, bools and arrays.")
            codes.append(c)
            types.append(t)
        ret = None
        for g in cands:
            G = self.funcs[g]
            if self._threaded(G):
                self.refuse(n, "a call through `%s` of `%s`, which changes an object passed to it" % (name, g),
                            "The translation gives the changed object back only at a direct call (classes.py); call `%s` "
                            "directly." % g)
            if len(G.params) != len(codes):
                self.refuse(n, "a call through `%s` with %d arguments, where `%s` takes %d" % (name, len(codes), g, len(G.params)),
                            "Pass every parameter (Viv fills no defaults through a function value).")
            for j, t in enumerate(types):
                if t.kind == "unknown":
                    continue
                jt = P.join(G.param_types[j], t)
                if jt is None:
                    self.refuse(n.args[j], "`%s` called with %s for `%s`, and elsewhere with %s" % (
                        g, self._kind_words(t), G.params[j], self._kind_words(G.param_types[j])),
                        "Write two functions, one per kind of argument.")
                G.param_types[j] = jt
            gr = G.ret or G.prev_ret
            if gr is not None and gr.kind == "tuple":
                self.refuse(n, "a call through `%s` of `%s`, which returns %d values" % (name, g, len(gr.elts)),
                            "Return one value.")
            ret = P.join(ret, gr) if gr is not None else ret
            if gr is not None and ret is None:
                self.refuse(n, "a call through `%s` of defs that return different types" % name, "Pass defs that return one type.")
        return "%s(%s)" % (self.vname(name), ", ".join(codes)), ret or P.UNK, 100

    def _effect_names(self):
        """The Python names whose Viv call may have an effect (print, write into an array): the
        program's defs, and the function parameters of the current def."""
        out = list(self.funcs)
        f = self.scope
        if f is not None:
            for p in f.params:
                if self._fn_param_index(p) is not None:
                    out.append(p)
        return out

    # ------------------------------------------------ default values
    def _default_ok(self, f, p, call):
        """A default is evaluated once, when the def runs; the translation evaluates it at each
        call, in the caller's scope. The same only for a number, or a module constant bound once
        before the def, which no scope of the caller hides."""
        d = f.defaults[p]
        if is_const_default(d):
            return
        once = self._module_once()
        for x in ast.walk(d):
            if isinstance(x, ast.Call) or isinstance(x, (ast.Lambda, ast.ListComp, ast.GeneratorExp, ast.Dict, ast.List)):
                break
            if isinstance(x, ast.Name):
                if x.id in self.np or x.id in self.math:
                    continue
                pos = once.get(x.id)
                hidden = self._comp_get(x.id) is not None or (self.scope is not None and x.id in self.scope.locals)
                if pos is None or pos >= (f.node.lineno, f.node.col_offset) or hidden:
                    break
        else:
            return
        self.refuse(call, "a call that leaves out `%s`, whose default `%s` Python computed when `%s` was defined"
                    % (p, ast.get_source_segment(self.src, d) or "...", f.name),
                    "Python evaluates a default once, at the def; the translation would read it at the call. "
                    "Pass `%s` here, or make the default a number." % p)

    def _module_once(self):
        """Module names bound exactly once at module level, never rebound by a def (`global`):
        name -> where."""
        m = getattr(self, "_once", None)
        if m is not None:
            return m
        counts, where = {}, {}
        for n in walk_scope([s for s in self.tree.body if not isinstance(s, (ast.FunctionDef, ast.ClassDef))]):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                counts[n.id] = counts.get(n.id, 0) + 1
                where[n.id] = _pos(n)
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Global):
                for g in n.names:
                    counts[g] = counts.get(g, 0) + 2
        self._once = {k: where[k] for k, c in counts.items() if c == 1}
        return self._once

    # ------------------------------------------------ min, max, sorted, next
    def _funcs_builtin(self, n, name):
        """The builtins translated here rather than by a rule: min and max of a generator or with
        key= or default=, sorted with key= or reverse= or of a generator, next of a generator.
        None when the rule should take the call (decided by the syntax, before translating)."""
        if name in ("min", "max") and self._no_def(name):
            if n.keywords or len(n.args) == 1 and isinstance(n.args[0], (ast.GeneratorExp, ast.ListComp)) \
                    or len(n.args) == 1 and _is_call(n.args[0], "range"):
                return self._minmax(n, name)
        if name == "sorted" and self._no_def(name):
            r = self._str_sorted(n)  # strings (strs.py)
            if r is not None:
                return r
            if n.keywords or len(n.args) == 1 and (isinstance(n.args[0], ast.GeneratorExp) or _is_call(n.args[0], "range")):
                return self._sorted(n)
            if len(n.args) == 1 and not isinstance(n.args[0], ast.Starred):
                # a list of rows (compared as lists) is sorted here; anything else by the rule,
                # which reads the argument translated once
                r = self.expr(n.args[0])
                if r[1].is_arr() and r[1].rank == 2 and r[1].container == "list":
                    return self._sorted(n, r)
                self._xmemo[id(n.args[0])] = r
        if name == "len" and self._no_def(name):
            r = self._str_set_len(n)  # len(set(s)) of text (strs.py)
            if r is not None:
                return r
        if name == "next" and self._no_def(name) and n.args and isinstance(n.args[0], ast.GeneratorExp):
            return self._next(n)
        if name == "list" and self._no_def(name) and len(n.args) == 1 and not n.keywords and \
                any(_is_call(n.args[0], f) for f in ("range", "zip", "enumerate", "reversed")):
            # list(range(...)), list(zip(...)), list(enumerate(...)): the comprehension of its items
            it = n.args[0]
            k = len(it.args) if _is_call(it, "zip") else (2 if _is_call(it, "enumerate") else 1)
            names = ["_py2viv_item%d" % j for j in range(k)]
            tgt = ast.Name(id=names[0], ctx=ast.Store()) if k == 1 else \
                ast.Tuple(elts=[ast.Name(id=x, ctx=ast.Store()) for x in names], ctx=ast.Store())
            elt = ast.Name(id=names[0], ctx=ast.Load()) if k == 1 else \
                ast.Tuple(elts=[ast.Name(id=x, ctx=ast.Load()) for x in names], ctx=ast.Load())
            lc = ast.ListComp(elt=elt, generators=[ast.comprehension(target=tgt, iter=it, ifs=[], is_async=0)])
            ast.copy_location(lc, n)
            ast.fix_missing_locations(lc)
            return self.expr(lc)
        return None

    def _as_gen(self, it, node):
        """An iterable as a one-clause generator of its items."""
        nm = "_py2viv_item"
        g = ast.GeneratorExp(elt=ast.Name(id=nm, ctx=ast.Load()),
                             generators=[ast.comprehension(target=ast.Name(id=nm, ctx=ast.Store()), iter=it, ifs=[], is_async=0)])
        ast.copy_location(g, it)
        ast.fix_missing_locations(g)
        return g

    def _minmax(self, n, name):
        """max(iterable, key=f, default=d), min(...), max(a, b, key=f): Python's order -- the
        first item, replaced by each later one whose key is greater (for min, smaller); an empty
        iterable raises ValueError unless there is a default."""
        kw = {}
        for k in n.keywords:
            if k.arg not in ("key", "default") or k.arg in kw:
                self.refuse(n, "%s with the keyword `%s`" % (name, k.arg), "Python raises TypeError; pass key= or default=.")
            kw[k.arg] = k.value
        key = kw.get("key")
        if key is not None and isinstance(key, ast.Constant) and key.value is None:
            key = None
        if any(isinstance(a, ast.Starred) for a in n.args):
            self.refuse(n, "a starred argument", "Pass the values one by one.")
        if not n.args:
            self.refuse(n, "%s of nothing" % name, "Python raises TypeError.")
        op = ">" if name == "max" else "<"
        M, K, S = self.new_tmp("m"), self.new_tmp("k"), self.new_tmp("f")
        state = {"t": None, "kt": None}

        def item(code, t, lines, ind, at):
            if t.kind == "bool" or t.kind in ("cnum", "carr", "str", "tuple", "none", "func") or t.is_arr() and t.rank != 1:
                self.refuse(at, "%s of %s" % (name, self._kind_words(t) + "s" if t.kind == "bool" else self._kind_words(t)),
                            "Python compares numbers (and, with a key, returns the item itself); pass numbers.")
            if t.is_arr() and key is None:
                self.refuse(at, "%s of rows without a key" % name, "Python compares the rows as a whole; pass key=.")
            jt = P.join(state["t"], t)
            if jt is None:
                self.refuse(at, "%s of items of different kinds" % name, "Pass items of one kind.")
            state["t"] = jt
            if not P.re.fullmatch(r"[A-Za-z_]\w*", code):
                tmp = self.new_tmp("x")
                lines.append(ind + "%s = %s" % (tmp, code))
                code = tmp
            if key is None:
                kc, kt = code, t
            else:
                saved = self.pre
                self.pre = []
                try:
                    kc, kt, _ = self._apply_fn(key, [(code, t)], at)
                finally:
                    got = self.pre
                    self.pre = saved
                lines.extend(ind + x for x in got)
                if kt.kind == "bool":
                    kc = "%s(%s)" % (self.use("py_b01"), kc)
                    kt = P.NUM_I
                if kt.kind != "num":
                    self.refuse(key, "a key that gives %s" % self._kind_words(kt), "A key gives a number.")
                if not P.re.fullmatch(r"[A-Za-z_]\w*", kc):
                    tmp = self.new_tmp("y")
                    lines.append(ind + "%s = %s" % (tmp, kc))
                    kc = tmp
            state["kt"] = kt
            lines.append(ind + "if not %s {" % S)
            lines.append(ind + "    %s = %s" % (M, code))
            if key is not None:
                lines.append(ind + "    %s = %s" % (K, kc))
            lines.append(ind + "    %s = true" % S)
            lines.append(ind + "}")
            lines.append(ind + "else if %s %s %s {" % (kc, op, K if key is not None else M))
            lines.append(ind + "    %s = %s" % (M, code))
            if key is not None:
                lines.append(ind + "    %s = %s" % (K, kc))
            lines.append(ind + "}")

        lines = []
        if len(n.args) > 1:
            if "default" in kw:
                self.refuse(n, "%s with default= and several arguments" % name, "Python raises TypeError.")
            for a in n.args:
                c, t, _ = self._sub(a, lines, "")
                item(c, t, lines, "", a)
        else:
            it = n.args[0]
            g = it if isinstance(it, ast.GeneratorExp) else self._as_gen(it, n)
            stack = []
            env = self._comp_push()
            try:
                ind = self._open_gens(g.generators, g, self._comp_binder(env), lines, "", stack)
                c, t, _ = self._sub(g.elt, lines, ind)
                item(c, t, lines, ind, g.elt)
            finally:
                self._comp_pop()
            self._close(lines, ind, stack)
        t = state["t"]
        init = ["%s = false" % S, "%s = %s" % (M, "zeros(0)" if t is not None and t.is_arr() else "0")]
        if key is not None:
            init.append("%s = 0" % K)
        after = ["if not %s {" % S]
        if "default" in kw:
            dc, dt, _ = self._sub(kw["default"], after, "    ")
            jt = P.join(t, dt)
            if jt is None or dt.kind != "num" and not dt.is_arr():
                self.refuse(kw["default"], "a default of another kind than the items", "Give a default of the items' kind.")
            if t is not None and t.kind == "num" and dt.kind == "num" and jt.py == "mixed":
                pass  # an int or a float, whichever it returns: typed as such
            t = jt
            after.append("    %s = %s" % (M, dc))
        else:
            after.append("    " + self._host_stop("%s_empty" % name, n))
        after.append("}")
        self.pre.extend(init + lines + after)
        return M, t or P.UNK, 100

    def _next(self, n):
        """next(generator), next(generator, default): its first item; StopIteration (or the
        default) when it has none."""
        if n.keywords or len(n.args) not in (1, 2):
            self.refuse(n, "next with these arguments", "Pass a generator expression and, at most, a default.")
        g = n.args[0]
        if len(n.args) == 2:
            d = n.args[1]
            if any(isinstance(x, ast.Call) for x in ast.walk(d)):
                self.refuse(d, "a default of next that calls a function",
                            "Python evaluates it before the generator runs; assign it to a variable first.")
        R, F = self.new_tmp("n"), self.new_tmp("f")
        lines, stack = [], []
        env = self._comp_push()
        try:
            ind = self._open_gens(g.generators, g, self._comp_binder(env), lines, "", stack)
            c, t, _ = self._sub(g.elt, lines, ind)
        finally:
            self._comp_pop()
        if t.kind not in ("num", "bool") and not (t.kind == "arr" and t.rank == 1):
            self.refuse(g.elt, "next of a generator of %s" % self._kind_words(t), "Generate numbers.")
        lines.append(ind + "%s = %s" % (R, c))
        lines.append(ind + "%s = true" % F)
        lines.append(ind + "break")
        self._close(lines, ind, stack, brk=F)
        zero = {"bool": "false"}.get(t.kind, "zeros(0)" if t.is_arr() else "0")
        after = ["if not %s {" % F]
        if len(n.args) == 2:
            dc, dt, _ = self._sub(n.args[1], after, "    ")
            jt = P.join(t, dt)
            if jt is None:
                self.refuse(n.args[1], "a default of another kind than the items (%s)" % self._kind_words(dt),
                            "Give a default of the items' kind.")
            t = jt
            after.append("    %s = %s" % (R, dc))
        else:
            after.append("    " + self._host_stop("stop_iteration", n))
        after.append("}")
        self.pre.extend(["%s = %s" % (R, zero), "%s = false" % F] + lines + after)
        return R, t, 100

    def _sort_args(self, n, name, first_pos=True):
        kw = {}
        for k in n.keywords:
            if k.arg not in ("key", "reverse") or k.arg in kw:
                self.refuse(n, "%s with the keyword `%s`" % (name, k.arg), "Python raises TypeError; pass key= or reverse=.")
            kw[k.arg] = k.value
        key = kw.get("key")
        if key is not None and isinstance(key, ast.Constant) and key.value is None:
            key = None
        rev = "0"
        if "reverse" in kw:
            r = kw["reverse"]
            if isinstance(r, ast.Constant) and isinstance(r.value, bool):
                rev = "1" if r.value else "0"
            elif isinstance(r, ast.Constant) and isinstance(r.value, int):
                rev = "1" if r.value else "0"
            else:
                c, t, _ = self.expr(r)
                if t.kind == "bool":
                    c = "%s(%s)" % (self.use("py_b01"), c)
                elif not (t.kind == "num" and t.py == "int"):
                    self.refuse(r, "reverse= of %s" % self._kind_words(t), "Pass True or False.")
                rev = c
        return key, rev

    def _sort_perm(self, items, t, key, rev, node, lines):
        """The code of a stable permutation that sorts items (a vector, or a matrix's rows) as
        Python's sorted does: by key (or the items), compared with `<`; with reverse, descending
        and still stable. A tuple key, or rows compared as a whole, is lexicographic: stable
        sorts by each column from the last to the first."""
        n = self.new_tmp("n")
        lines.append("%s = len(%s)" % (n, items))
        cols = []
        if key is None:
            if t.rank == 1:
                cols = [items]
            else:
                if t.container != "list":
                    self.refuse(node, "sorting the rows of a NumPy matrix without a key",
                                "Python compares NumPy rows with `<` and raises ValueError; pass key=.")
                # rows of a list: compared as lists, first entry first
                w = self.new_tmp("c")
                lines.append("%s = %s(%s)" % (w, self.use("py_ncols"), items))
                cols = None
                colsrc = (items, w)
        else:
            k = self.new_tmp("i")
            x = self.new_tmp("x")
            body = []
            if t.rank == 1:
                ec, et = self._elem_of(items, k, t)
            else:
                ec, et = "slab(%s, %s)" % (items, k), P.T(t.kind, 1, t.py, t.container, elts=t.elts)
            body.append("    %s = %s" % (x, ec))
            parts = key.body.elts if isinstance(key, ast.Lambda) and isinstance(key.body, ast.Tuple) else None
            saved = self.pre
            self.pre = []
            try:
                if parts is not None:
                    kvals = []
                    for j, part in enumerate(parts):
                        lam = ast.copy_location(ast.Lambda(args=key.args, body=part), key)
                        kc, kt, _ = self._apply_fn(lam, [(x, et)], node)
                        kvals.append((kc, kt, part))
                else:
                    kc, kt, _ = self._apply_fn(key, [(x, et)], node)
                    kvals = [(kc, kt, key)]
            finally:
                got = self.pre
                self.pre = saved
            body.extend("    " + y for y in got)
            for j, (kc, kt, at) in enumerate(kvals):
                if kt.kind == "bool":
                    kc = "%s(%s)" % (self.use("py_b01"), kc)
                elif kt.kind != "num":
                    self.refuse(at, "a sort key that gives %s" % self._kind_words(kt),
                                "A key gives a number, or a tuple of numbers.")
                col = self.new_tmp("K")
                lines.append("%s = zeros(%s)" % (col, n))
                body.append("    %s[%s] = %s" % (col, k, kc))
                cols.append(col)
            lines.append("for %s to %s {" % (k, n))
            lines.extend(body)
            lines.append("}")
        p = self.new_tmp("p")
        lines.append("%s = %s(%s)" % (p, self.use("py_iota"), n))
        if cols is None:
            items_, w = colsrc
            j = self.new_tmp("j")
            lines.append("for %s from 0 to %s by 1 reversed {" % (j, w))
            lines.append("    %s = %s(%s(%s, %s), %s, %s)" % (p, self.use("py_msort"), self.use("py_col"), items_, j, p, rev))
            lines.append("}")
        else:
            for col in reversed(cols):
                lines.append("%s = %s(%s, %s, %s)" % (p, self.use("py_msort"), col, p, rev))
        return p

    def _sorted(self, n, done=None):
        """sorted(iterable, key=f, reverse=r): a new list, stably sorted (see _sort_perm)."""
        if len(n.args) != 1 or isinstance(n.args[0], ast.Starred):
            self.refuse(n, "sorted with %d positional arguments" % len(n.args), "Python raises TypeError; pass one iterable.")
        key, rev = self._sort_args(n, "sorted")
        it = n.args[0]
        if isinstance(it, ast.GeneratorExp):
            it = ast.copy_location(ast.ListComp(elt=it.elt, generators=it.generators), it)
        elif _is_call(it, "range"):
            it = ast.copy_location(ast.ListComp(elt=self._as_gen(it, n).elt, generators=self._as_gen(it, n).generators), it)
        code, t, _ = done if done is not None else self.expr(it)
        if t.kind == "str":
            self.refuse(it, "sorting a string", "Viv strings are not values.")
        if not t.is_arr() or t.kind == "carr" or t.rank > 2 or t.container == "shape":
            self.refuse(it, "sorting %s" % self._kind_words(t), "Sort a vector, or a list of rows.")
        if t.py == "bool" and key is None:
            self.refuse(it, "sorting Bools", "Python keeps them Bools; sort numbers.")
        if t.py == "mixed" and t.rank == 1:
            self.refuse(it, "sorting a list mixing ints and floats", "Python keeps each as it is; make them all floats.")
        items = code
        lines = []
        if t.rank == 2 and t.name != "tuple" and not isinstance(it, ast.ListComp):
            self._fresh_row(it, n)  # the sorted list shares the rows (a shallow copy)
        if not P.re.fullmatch(r"[A-Za-z_]\w*", items):
            items = self.new_tmp("s")
            lines.append("%s = %s" % (items, code))
        p = self._sort_perm(items, t, key, rev, n, lines)
        R = self.new_tmp("l")
        if t.rank == 1:
            lines.append("%s = take(%s, %s)" % (R, items, p))
        else:
            lines.append("%s = %s(%s, %s)" % (R, self.use("py_rowsat"), items, p))
        self.pre.extend(lines)
        return R, P.T(t.kind, t.rank, t.py, "list", elts=t.elts, name=t.name), 100

    # ------------------------------------------------ a for loop over a generator
    def _funcs_for(self, s):
        """A for loop py2viv.py's own translation does not take: over a generator expression
        (map and filter become one), over enumerate(a, start), over reversed(a), and a target
        that unpacks rows. True when translated here."""
        it, tgt = s.iter, s.target
        mine = isinstance(it, ast.GeneratorExp)
        if _is_call(it, "reversed") and not (it.args and _is_call(it.args[0], "range")):
            mine = True
        if _is_call(it, "enumerate") and (len(it.args) > 1 or it.keywords or
                                           isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2 and not isinstance(tgt.elts[1], ast.Name)):
            mine = True
        if _is_call(it, "zip") and (isinstance(tgt, ast.Tuple) and any(not isinstance(e, ast.Name) for e in tgt.elts)
                                    or any(isinstance(a, ast.GeneratorExp) or _is_call(a, "reversed") for a in it.args)):
            mine = True
        if isinstance(tgt, (ast.Tuple, ast.List)) and not (_is_call(it, "enumerate") or _is_call(it, "zip")
                                                           or isinstance(it, (ast.Tuple, ast.List))):
            mine = True
        if not mine:
            return False
        if s.orelse:
            self.refuse(s, "`for ... else`", "Use a flag variable set before `break`.")
        lines, stack = [], []
        ind = self._open_clause(it, tgt, s, self._stmt_binder(s), lines, "", stack)
        if sum(1 for k in stack if k == "loop") > 1:
            for st in s.body:
                for m in _loop_level(st):
                    if isinstance(m, ast.Break):
                        self.refuse(m, "`break` in a loop over a generator with several `for` clauses",
                                    "Set a flag and test it; Viv's break leaves one loop.")
        self.flush_pre(s)
        for x in lines:
            self.emit(x, s)
        depth = len(stack)
        self.indent += depth
        self._block(s.body)
        self.indent -= depth
        for k in range(depth):
            self.emit("    " * (depth - 1 - k) + "}", s)
        return True


def _loop_level(st):
    """The nodes of a statement outside any loop or def nested in it."""
    stack = [st]
    while stack:
        n = stack.pop()
        yield n
        for c in ast.iter_child_nodes(n):
            if isinstance(c, (ast.For, ast.While, ast.FunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            stack.append(c)


class ElementTypes:
    """Items that keep their own types: a tuple's entries (a list of tuples is a matrix whose
    columns do), and a list literal of ints beside floats. `T.elts` holds the types; printing
    such a value, or storing into it, is refused (Python prints a tuple with parentheses, a
    list of ints and floats each as it is; a tuple cannot be written)."""

    def _funcs_literal(self, n):
        elts = n.elts
        if any(isinstance(e, ast.Starred) for e in elts):
            return None
        if all(isinstance(e, ast.Tuple) for e in elts) and elts:
            # rows of tuples: a matrix with typed columns
            w = len(elts[0].elts)
            if w == 0 or any(len(e.elts) != w for e in elts):
                return None
            rows, cols = [], [None] * w
            for e in elts:
                codes = []
                for j, x in enumerate(e.elts):
                    c, t, _ = self.expr(x)
                    if t.kind == "bool":
                        c = {"true": "1", "false": "0"}.get(c) or "%s(%s)" % (self.use("py_b01"), c)
                    elif t.kind != "num":
                        self.refuse(x, "a tuple holding %s in a list" % self._kind_words(t),
                                    "A list of tuples py2viv translates holds tuples of numbers.")
                    cols[j] = t if cols[j] is None else P.join(cols[j], t)
                    codes.append(c)
                rows.append("[%s]" % ", ".join(codes))
            py = "float" if any(t.py in ("float", "mixed") for t in cols) else "int"
            return "[%s]" % ", ".join(rows), P.T("arr", 2, py, "list", elts=cols, name="tuple"), 100
        if any(isinstance(e, (ast.List, ast.Tuple, ast.ListComp)) for e in elts):
            return None
        items = [self.expr(e) for e in elts]
        types = [t for _, t, _ in items]
        tup = isinstance(n, ast.Tuple) and all(t.kind in ("num", "bool") for t in types)
        if tup or all(t.kind == "num" for t in types) and {t.py for t in types} >= {"int"} and \
                any(t.py in ("float", "mixed") for t in types):
            # a tuple of numbers (each entry keeps its type; Python prints it with parentheses),
            # or a list of ints beside floats
            codes = []
            for c, t, _ in items:
                codes.append(c if t.kind == "num" else ({"true": "1", "false": "0"}.get(c) or "%s(%s)" % (self.use("py_b01"), c)))
            py = "float" if any(t.py in ("float", "mixed") for t in types) else ("int" if any(t.kind == "num" for t in types) else "bool")
            return "[%s]" % ", ".join(codes), P.T("arr", 1, py, "list", elts=types, name="tuple" if tup else None), 100
        # the plain literal: its items were translated once, and are read again from the memo
        self._xmemo.update({id(e): r for e, r in zip(elts, items)})
        return None

    def _elts_item(self, bt, idx):
        """The type of one item of a value whose items keep their types: its own for an int
        literal index, else the type they share (an int on some paths and a float on others
        when they differ)."""
        k = None
        if isinstance(idx, ast.Constant) and type(idx.value) is int:
            k = idx.value
        elif isinstance(idx, ast.UnaryOp) and isinstance(idx.op, ast.USub) and isinstance(idx.operand, ast.Constant) \
                and type(idx.operand.value) is int:
            k = -idx.operand.value
        if k is not None and -len(bt.elts) <= k < len(bt.elts):
            t = bt.elts[k]
        else:
            t = join_all(bt.elts) or P.NUM_MIXED
        return t  # a Bool item is read back as a Bool (py2viv.py, _e_Subscript)

    def _elts_store(self, tgt, bt, t):
        if bt.name == "tuple":
            self.refuse(tgt, "a store into a tuple", "Python raises TypeError (a tuple cannot change); build a new one.")
        self.refuse(tgt, "a store into a list of ints and floats", "Python keeps each item's type; make them all floats.")

    def _elts_print(self, node, t):
        if t.name == "tuple":
            self.refuse(node, "printing a tuple, or a list of tuples", "Python writes tuples with parentheses; print the numbers.")
        if any(e.kind == "bool" for e in t.elts or ()):
            self.refuse(node, "printing a list of Bools beside numbers", "Python writes each as it is (True or 1); print them one by one.")
        self.refuse(node, "printing a list of ints and floats", "Python writes each as it is (3 or 3.0); make them all floats.")


class ListMethods:
    """A Python list's methods and operators, for the lists py2viv can follow (see the rules in
    rules/numpy.toml, section "Python lists")."""

    def _grown_type(self, name):
        """The type of a list py2viv grows, from its appends: numbers of one type, or rows."""
        key = (self._scope_of(name), name)
        prev_py, prev_shape = getattr(self, "grow_prev", ({}, {}))
        shp = self.grow_shape.get(key) or (prev_shape.get(key) if key not in self.grow_py else None)
        if shp is not None and shp[0] == "rows":
            return P.T("arr", 2, shp[1], "list", elts=list(shp[2]) if shp[2] else None, name=shp[3])
        py = self.grow_py.get(key) or prev_py.get(key, "float")
        if py == "str":
            return P.STRS  # a list of strings (strs.py)
        return P.T("arr", 1, py, "list")

    def _row_shape(self, x, t, what="appending"):
        """("rows", py, column types, "tuple" or None) of a row put into a list, refusing a row
        another name can reach (a list of lists shares its rows in Python; Viv copies them)."""
        if not (t.kind == "arr" and t.rank == 1):
            self.refuse(x, "%s %s to a list" % (what, self._kind_words(t)), "A list py2viv grows holds numbers, or rows of numbers.")
        tup = t.name == "tuple"
        if not tup:
            self._fresh_row(x, x)
        if t.py == "mixed" and not t.elts or t.py == "bool" and not tup:
            self.refuse(x, "a row of %s in a list" % ("Bools" if t.py == "bool" else "ints and floats"),
                        "Make the row's entries all floats.")
        elts = tuple(t.elts) if t.elts else None
        # a NumPy array put into a list stays an array there: Python prints it `array([...])`
        # and computes with it as NumPy does (RULES.md, Fixed holes: list-of-arrays)
        return ("rows", t.py, elts, "tuple" if tup else ("ndrows" if t.container == "nd" else None))

    def _grow_row(self, lst, x, t, code):
        """lst.append(row): the row goes on as a new row of Viv's matrix (a tuple as a row whose
        entries keep their types); rows of different lengths stop the run."""
        key = (self._scope_of(lst.id), lst.id)
        if key in self.grow_py:
            self.refuse(x, "a row appended to a list of numbers", "Python keeps both; keep numbers and rows in two lists.")
        shape = self._row_shape(x, t)
        old = self.grow_shape.setdefault(key, shape)
        if old != shape:
            self.refuse(x, "rows of another kind appended to a list of rows (%s)" % ("tuples and lists" if old[3] != shape[3] else
                        "ints and floats" if old[1] != shape[1] else "other entry types"),
                        "Python keeps each as it is (and prints them so); append rows of one kind.")
        self.set_var(lst.id, self._grown_type(lst.id), x)
        n = self.vname(lst.id)
        row = code
        if not P.re.fullmatch(r"[A-Za-z_]\w*", row):
            row = self.new_tmp("w")
            self.pre.append("%s = %s" % (row, code))
        lines = []
        self._ragged_check(n, row, lines, "")
        self.pre.extend(lines)
        return "%s = append(%s, %s)" % (n, n, row)

    def _list_method(self, ctx, m, views):
        """The hook list_method: lst.pop(i), insert, extend, remove, clear, reverse, sort(key=,
        reverse=), index, count. The methods that change the list rebind its name (the list is
        its own: `a.grows`); sort writes back in place."""
        node = ctx.node
        a = views[0]
        stmt = ctx.rule.context == "stmt"
        if m == "sort":
            return self._list_sort(node, a, views[1], views[2])
        at = a.t
        lst = a.node
        code = a.code
        if not (at.is_arr() and at.container == "list") or at.kind == "carr":
            self.refuse(node, "`.%s` of %s" % (m, self._kind_words(at)), "A NumPy array has no .%s (AttributeError); use a list." % m)
        rows = at.rank == 2
        if m in ("index", "count"):
            x = views[1]
            xc, xt = x.code, x.t
            if xt.kind == "bool":
                xc = "%s(%s)" % (self.use("py_b01"), xc)
            elif xt.kind != "num":
                self.refuse(x.node, "`.%s` of %s" % (m, self._kind_words(xt)), "Look for a number.")
            return "%s(%s, %s)" % (self.use("py_indexv" if m == "index" else "py_countv"), code, xc), P.NUM_I, 100
        if not isinstance(lst, ast.Name):
            self.refuse(node, "`.%s` of a list that is not a plain name" % m, "Call it on the list's name.")
        key = (self._scope_of(lst.id), lst.id)
        self._seed_param_list(lst.id, key)
        n = self.vname(lst.id)
        if m == "pop":
            i = views[1]
            if i is not None and i.given:
                it = i.t
                if it.kind != "num" or it.py != "int":
                    self.refuse(i.node, "list.pop with an index that is %s" % ("a Bool" if it.kind == "bool" else "not an int"),
                                "Python raises TypeError (or pops True as 1); pass an int.")
                ic = i.code
            else:
                ic = "-1"
            k = self.new_tmp("k")
            self.pre.append("%s = %s(len(%s), %s)" % (k, self.use("py_popk"), n, ic))
            drop = "%s = %s(%s, %s)" % (n, self.use("py_dropm" if rows else "py_dropv"), n, k)
            if stmt:
                return drop, None, 100
            x = self.new_tmp("x")
            if rows:
                self.pre.append("%s = slab(%s, %s)" % (x, n, k))
                xt = P.row_type(at)
            else:
                ec, xt = self._elem_of(n, k, at)
                self.pre.append("%s = %s" % (x, ec))
            self.pre.append(drop)
            return x, xt, 100
        if m == "clear":
            return "%s = []" % n, None, 100
        if m == "reverse":
            return "%s = %s(%s)" % (n, self.use("py_revm" if rows else "py_revv"), n), None, 100
        if m == "remove":
            if rows:
                self.refuse(node, "list.remove of a row", "Find its index and pop it.")
            x = views[2]
            xc, xt = self._list_item(key, x, "removing")
            return "%s = %s(%s, %s)" % (n, self.use("py_removev"), n, xc), None, 100
        if m == "insert":
            i, x = views[1], views[2]
            it = i.t
            if it.kind != "num" or it.py != "int":
                self.refuse(i.node, "list.insert at an index that is not an int", "Python raises TypeError; pass an int.")
            k = self.new_tmp("k")
            self.pre.append("%s = %s(len(%s), %s)" % (k, self.use("py_insk"), n, i.code))
            if rows or x.t.is_arr():
                shape = self._row_shape(x.node, x.t, "inserting")
                if self.grow_shape.setdefault(key, shape) != shape or key in self.grow_py:
                    self.refuse(x.node, "a row of another kind inserted into a list", "Insert rows of one kind.")
                self.set_var(lst.id, self._grown_type(lst.id), node)
                return "%s = %s(%s, %s, %s)" % (n, self.use("py_insm"), n, k, x.code), None, 100
            xc, xt = self._list_item(key, x, "inserting")
            return "%s = %s(%s, %s, %s)" % (n, self.use("py_insv"), n, k, xc), None, 100
        if m == "extend":
            return self._list_extend(node, key, lst, n, at, views[2])
        self.refuse(node, "list.%s" % m, "Not translated.")

    def _list_item(self, key, x, what):
        """A number put into (or looked for in) a list of numbers: of the list's type (Python keeps
        an int an int and a float a float, and prints them so)."""
        xt = x.t
        if xt.kind == "bool" or xt.kind != "num":
            self.refuse(x.node, "%s %s" % (what, self._kind_words(xt)), "A list py2viv changes holds numbers of one type.")
        if xt.py == "mixed":
            self.refuse(x.node, "%s a number that is an int on some paths and a float on others" % what, "Convert it with float(...).")
        if what != "removing":
            if self.grow_py.setdefault(key, xt.py) != xt.py or self.grow_shape.get(key, ("num",))[0] != "num":
                self.refuse(x.node, "a list of ints and floats", "Python prints each as it is; make them all floats.")
        return x.code, xt

    def _list_extend(self, node, key, lst, n, at, x):
        xn = x.node
        if isinstance(xn, ast.GeneratorExp) or _is_call(xn, "range"):
            g = xn if isinstance(xn, ast.GeneratorExp) else self._as_gen(xn, node)
            xn = ast.copy_location(ast.ListComp(elt=g.elt, generators=g.generators), xn)
            xc, xt, _ = self.expr(xn)
        else:
            xc, xt = x.code, x.t
        if xt.kind == "str":
            self.refuse(x.node, "extending a list by a string", "Viv strings are not values.")
        if not xt.is_arr() or xt.kind == "carr" or xt.container == "shape":
            self.refuse(x.node, "extending a list by %s" % self._kind_words(xt), "Extend it by a list of numbers.")
        if xt.rank == 1 and at.rank == 1:
            if xt.py in ("bool", "mixed") or xt.elts:
                self.refuse(x.node, "extending a list by %s" % ("Bools" if xt.py == "bool" else "ints and floats"),
                            "Python keeps each item's type; extend by numbers of one type.")
            if self.grow_py.setdefault(key, xt.py) != xt.py or self.grow_shape.get(key, ("num",))[0] != "num":
                self.refuse(x.node, "a list of ints and floats", "Python prints each as it is; make them all floats.")
            self.set_var(lst.id, P.T("arr", 1, xt.py, "list"), node)
            return "%s = extend(%s, %s)" % (n, n, xc), None, 100
        if xt.rank == 2 and isinstance(xn, (ast.List, ast.ListComp)):
            # rows of a new list: each goes on as a row
            shape = ("rows", xt.py, tuple(xt.elts) if xt.elts else None, xt.name)
            if key in self.grow_py or self.grow_shape.setdefault(key, shape) != shape:
                self.refuse(x.node, "rows of another kind added to a list", "Add rows of one kind.")
            self.set_var(lst.id, self._grown_type(lst.id), node)
            src = xc
            if not P.re.fullmatch(r"[A-Za-z_]\w*", src):
                src = self.new_tmp("s")
                self.pre.append("%s = %s" % (src, xc))
            k, w = self.new_tmp("i"), self.new_tmp("w")
            lines = ["for %s to len(%s) {" % (k, src), "    %s = slab(%s, %s)" % (w, src, k)]
            self._ragged_check(n, w, lines, "    ")
            lines += ["    %s = append(%s, %s)" % (n, n, w), "}"]
            return "\n".join(lines), None, 100
        self.refuse(x.node, "extending a list of %s by %s" % ("rows" if at.rank == 2 else "numbers", self._kind_words(xt)),
                    "A list of rows shares its rows with the list it came from in Python; extend by new rows (a list literal "
                    "or a comprehension).")

    def _list_sort(self, node, a, key, rev):
        """lst.sort(key=f, reverse=r): stably, in place (every name holding the list sees it)."""
        at = a.t
        if not (at.is_arr() and at.container == "list") or at.kind == "carr":
            self.refuse(node, "`.sort` with key= or reverse= of %s" % self._kind_words(at),
                        "NumPy's sort takes neither (TypeError); use sorted(..., key=...) of a list.")
        fake = ast.copy_location(ast.Call(func=ast.Name(id="sorted", ctx=ast.Load()), args=[], keywords=node.keywords), node)
        keyn, revc = self._sort_args(fake, "list.sort")
        if at.py == "bool" and keyn is None:
            self.refuse(node, "sorting Bools", "Python keeps them Bools; sort numbers.")
        if at.py == "mixed" and at.rank == 1:
            self.refuse(node, "sorting a list mixing ints and floats", "Python keeps each as it is; make them all floats.")
        items = a.code
        lines = []
        p = self._sort_perm(items, at, keyn, revc, node, lines)
        self.pre.extend(lines)
        if at.rank == 1:
            return "%s(%s, take(%s, %s))" % (self.use("py_setall"), items, items, p), None, 100
        return "%s(%s, %s(%s, %s))" % (self.use("py_setall"), items, self.use("py_rowsat"), items, p), None, 100

    # ------------------------------------------------ `in` and `==` of lists
    def _funcs_in(self, n, op, lc, lt, right):
        """x in seq, x not in seq, for seq a list or a NumPy array (not a literal): Python's list
        compares each item with `==` (a NaN only as the same object: the run stops); NumPy's `in`
        is (a == x).any(). None when it is not one of these."""
        rc, rt, rp = self.expr(right)
        if not rt.is_arr() or rt.kind == "carr" or rt.container == "shape":
            if rt.kind == "str" or lt.kind == "str":
                self.refuse(n, "`in` of a string", "Viv strings are not values.")
            self.refuse(n, "`in` of %s" % self._kind_words(rt), "Test membership in a list or an array.")
        x = lc
        if lt.kind == "bool":
            x = "%s(%s)" % (self.use("py_b01"), lc)
        elif lt.kind != "num":
            self.refuse(n, "`in` with %s on the left" % self._kind_words(lt), "Look for a number.")
        if rt.container == "list":
            if rt.rank != 1:
                self.refuse(n, "`in` of a list of rows", "Python compares the rows as a whole; test the entries.")
            code = "%s(%s, %s)" % (self.use("py_inlist"), rc, x)
        else:
            # Viv's `x in a` on an array is NumPy's: (a == x).any()
            word = "not in" if isinstance(op, ast.NotIn) else "in"
            return "%s %s %s" % (self._paren(x, 100 if P.re.fullmatch(r"[\w.]+", x) else 50, 51), word, self._paren(rc, rp, 51))
        if isinstance(op, ast.NotIn):
            return "not %s" % code
        return code

    def _funcs_listeq(self, n, sym, lc, lt, rc, rt):
        """a == b, a != b of two Python lists (or tuples): the same length and every item equal,
        one Bool. A list and a tuple are never equal."""
        if (lt.name == "tuple") != (rt.name == "tuple"):
            return "false" if sym == "==" else "true"
        code = "%s(%s, %s)" % (self.use("py_listeq"), lc, rc)
        return code if sym == "==" else "not %s" % code

    def _grow_rows_literal(self, tgt, value, items, s):
        """lst = [r1, r2] of a list then grown by rows: an empty list and an append per row."""
        key = (self._scope_of(tgt.id), tgt.id)
        shape = None
        for e, (c, t, _) in zip(value.elts, items):
            sh = self._row_shape(e, t)
            if shape is not None and sh != shape:
                self.refuse(e, "rows of different kinds in one list", "Python keeps each as it is; make the rows alike.")
            shape = sh
        if key in self.grow_py or self.grow_shape.setdefault(key, shape) != shape:
            self.refuse(value, "rows of another kind than the list's appends", "Make the rows alike.")
        self.set_var(tgt.id, self._grown_type(tgt.id), s)
        self.flush_pre(s)
        n = self.vname(tgt.id)
        self.emit("%s = []" % n, s)
        for c, t, _ in items:
            self.emit("%s = append(%s, %s)" % (n, n, c), s)

    # ------------------------------------------------ text in a print
    def _print_text(self, a, parts, values):
        """A print argument that builds text: str(x), repr(x), and `+` of strings, f-strings and
        those. Its pieces are print parts (text, and values written as print writes them: str of a
        number or a list is what print writes). True when a is one."""
        if not self._is_text(a):
            return False
        self._text_parts(a, parts, values)
        return True

    def _is_text(self, a):
        if isinstance(a, ast.Constant) and isinstance(a.value, str) or isinstance(a, ast.JoinedStr):
            return True
        if isinstance(a, ast.Call) and isinstance(a.func, ast.Name) and a.func.id in ("str", "repr") \
                and self._no_def(a.func.id) and len(a.args) == 1 and not a.keywords:
            return True
        if isinstance(a, ast.BinOp) and isinstance(a.op, ast.Add):
            return (self._is_text(a.left) or self._is_text(a.right)) and all(
                self._is_text(x) or isinstance(x, ast.BinOp) and isinstance(x.op, ast.Add) for x in (a.left, a.right))
        return False

    def _text_parts(self, a, parts, values):
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            parts.append({"kind": "str", "text": a.value})
            return
        if isinstance(a, ast.JoinedStr):
            self._fstring(a, parts, values)
            return
        if isinstance(a, ast.Call):
            x = a.args[0]
            if isinstance(x, ast.Constant) and isinstance(x.value, str):
                parts.append({"kind": "str", "text": repr(x.value) if a.func.id == "repr" else x.value})
                if getattr(x, "_py2viv_subst", False):
                    parts[-1]["subst"] = True  # Python read a variable here (diff.py)
                return
            code, t, _ = self.expr(x, keep=True)  # an Int, a member (ints.py)
            self._print_value(x, code, t, a.func.id, None, parts, values)
            return
        if isinstance(a, ast.BinOp) and isinstance(a.op, ast.Add):
            for side in (a.left, a.right):
                if self._is_text(side):
                    self._text_parts(side, parts, values)
                    continue
                # a string computed another way ("[" + ", ".join(...) + "]"): one value (strs.py)
                code, t, _ = self.expr(side)
                if t.kind != "str":
                    self.refuse(a, "`+` of text and something that is not text", "Python raises TypeError; convert with str(...).")
                self._print_value(side, code, t, "str", None, parts, values)
            return
        self.refuse(a, "this text", "Print string literals, f-strings and numbers.")

    # ------------------------------------------------ a negative step
    def _funcs_rslice(self, n):
        """a[::-1], a[i:j:-k] of a vector (a literal negative step): its entries walked backward,
        a new array. Bounds as Python's slice.indices reads them for a negative step (py_rslice).
        A NumPy slice is a view: the view check applies as to any slice. NumPy sums a view with a
        negative step in the order of its indices, as its copy (checked on 909 random arrays of
        up to 30001 entries, steps -1 to -3, 2026-09-28)."""
        if not isinstance(n.value, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)) or isinstance(n.slice, ast.Tuple):
            return None
        sl = n.slice
        if not isinstance(sl, ast.Slice) or sl.step is None:
            return None
        st = sl.step
        if not (isinstance(st, ast.UnaryOp) and isinstance(st.op, ast.USub) and isinstance(st.operand, ast.Constant)
                and type(st.operand.value) is int and st.operand.value > 0):
            return None
        if isinstance(n.value, ast.Subscript) and any(isinstance(x, ast.Slice) for x in ast.walk(n.value.slice)):
            return None
        base, bt, bp = self.expr(n.value)
        if not bt.is_arr() or bt.rank != 1 or bt.kind == "carr" or bt.container == "shape":
            self._xmemo[id(n.value)] = (base, bt, bp)  # translated once: the slice's own path reads it
            return None
        parts = []
        for b in (sl.lower, sl.upper):
            if b is None:
                parts += ["0", "0"]
                continue
            c, t, _ = self.expr(b)
            if t.kind != "num" or t.py != "int":
                self.refuse(b, "a slice bound that is %s" % self._kind_words(t),
                            "Python takes only ints as slice bounds (a float raises TypeError); convert with int(...).")
            parts += ["1", c]
        elts = None
        if bt.elts:
            if sl.lower is None and sl.upper is None and st.operand.value == 1:
                elts = list(reversed(bt.elts))
            else:
                self.refuse(n, "a slice of a list of ints and floats", "Python keeps each item's type; make them all floats.")
        code = "%s(%s, %s, %s, %s, %s, %d)" % (self.use("py_rslice"), base, parts[0], parts[1], parts[2], parts[3],
                                               -st.operand.value)
        if bt.container == "nd":
            self._slice_in_place(n)
        return code, P.T(bt.kind, 1, bt.py, bt.container, elts=elts, name=bt.name), 100

    def _inside(self, node, ancestor):
        """Whether node lies inside ancestor (by the parent map)."""
        p = self.parents.get(node)
        while p is not None:
            if p is ancestor:
                return True
            p = self.parents.get(p)
        return False

    # ------------------------------------------------ nested unpacking
    def _nested_unpack(self, tgt, value, s):
        """(a, b), c = u, 2.0: every right side first (into temporaries, as Python evaluates them
        all before any store), then each target, a tuple of names unpacking a vector."""
        vals = []
        for v in value.elts:
            if isinstance(v, ast.Starred):
                self.refuse(v, "a starred value", "Write the values one by one.")
            c, t, _ = self.expr(v)
            if not P.re.fullmatch(r"-?\d+(\.\d+)?|true|false", c):
                tmp = self.new_tmp("u")
                self.pre.append("%s = %s" % (tmp, c))
                c = tmp
            vals.append((c, t))
        for (c, t), tg, v in zip(vals, tgt.elts, value.elts):
            if isinstance(tg, (ast.Tuple, ast.List)):
                if any(isinstance(x, (ast.Tuple, ast.List, ast.Starred)) for x in tg.elts):
                    self.refuse(tg, "an unpacking target nested two levels deep", "Unpack in two steps.")
                if not (t.kind == "arr" and t.rank == 1):
                    self.refuse(v, "unpacking %s" % self._kind_words(t), "Unpack a vector or a tuple.")
                if t.elts:
                    if len(t.elts) != len(tg.elts):
                        self.refuse(v, "unpacking %d values into %d names" % (len(t.elts), len(tg.elts)), "Give one name per value.")
                    types = list(t.elts)
                else:
                    et = P.BOOL if t.py == "bool" else (P.NUM_I if t.py == "int" else P.NUM_F)
                    types = [et] * len(tg.elts)
                self._unpack_into(tg.elts, c, types, s)
            elif isinstance(tg, ast.Starred):
                self.refuse(tg, "a starred unpacking target", "Unpack into as many names as there are values.")
            else:
                self._store(tg, c, t, s)
