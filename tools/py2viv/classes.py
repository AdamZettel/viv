# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""classes.py: Python classes, dataclasses, named tuples and dicts with literal keys, as Viv
records (py2viv, strict).

A Viv record is a value: assigning it, passing it and returning it copy its
numbers and share its arrays (an array field is one handle in every engine). A Python object is
a reference. The two agree exactly while no field of the object is rebound after construction:
then a copy and a second reference are indistinguishable, and a write into an array field goes
through the one shared array in both. This module translates

  - a class whose attributes are all set in `__init__` (first set at the top level of its
    body, before any read), plain or `@dataclass` (frozen or not), `typing.NamedTuple` and
    `collections.namedtuple`: a `record` and a constructor def of the same name, whose body is
    `__init__`'s with `self.f` as a local, and which returns the record literal;
  - methods: defs `C__m(self, ...)` taking the record first; `@property` (a def of `self`
    alone, called where the attribute is read) and `@staticmethod` (a def without `self`);
    class attributes that are number literals (module variables `C__K`);
  - a dict literal whose keys are string or int constants, read by constant keys: a record
    with one field per key, in insertion order; `len`, `in` with a constant, `.get`, `.copy`,
    iteration over the keys, values or items (unrolled, one copy of the body per key), and
    printing, all known at translation time;
  - printing (print, f-strings) of a dataclass, a named tuple or a dict, as CPython writes it
    (`P(x=1.5, y=2)`, `{'a': 1}`); a plain class whose `__str__` or `__repr__` returns an
    f-string prints through that f-string.

Fields that ARE rebound after construction (`self.n += 1` in a method, `p.x = v`, `d["k"] = v`)
are translated by rebinding the variable that holds the object to a new record
(`p = C { x: v, y: p.y }`). That is exact only if no other name, container or field holds the
same object, so for such a class (or dict shape) every use that could make a second reference
is refused: `q = p`, `return p` of a parameter or a module variable, an object stored in a
field, a list or a dict, the same object passed twice to a def that changes it. A def or
method that rebinds a field of an object passed to it (its `self`, or any parameter) returns
the new object along with its own results, and every call rebinds the caller's variable
(`p, r = C__step(p, dt)`): state threading. Such a call must stand as a statement, as the whole
right side of an assignment, or as a `return` value, so that Python's order of evaluation is
kept; the argument it changes must be a variable of the caller.

What stays refused (with the reason): inheritance, metaclasses, most dunder methods (operators,
`__eq__`, `__len__`, `__iter__`, ... are not translated; using the operator is refused where it
is used), identity (`is`), `isinstance`, attributes created outside `__init__`, `self` used as
a value inside `__init__`, None-valued attributes, linked structures (a class holding its own
objects), lists of objects, dicts with keys computed at run time, empty dicts grown by
assignment, deletion.
"""

import ast
import copy
import keyword
import re

# The translator's names (py2viv.py hands them over with install(); importing py2viv here would
# load a second copy of it when it runs as a script).
T = join = viv_name = FuncInfo = assigned_names = Refusal = None
NUM_I = NUM_F = NUM_MIXED = BOOL = UNK = NONE = STR = CNUM = None
tup = arr = rule_engine = node_pos = None


def install(ns):
    g = globals()
    for k in ("T", "join", "viv_name", "FuncInfo", "assigned_names", "Refusal", "NUM_I", "NUM_F", "NUM_MIXED",
              "BOOL", "UNK", "NONE", "STR", "CNUM", "tup", "arr", "rule_engine", "node_pos"):
        g[k] = ns[k]


# Dunder methods that change what creating, reading, writing or printing an object does: a
# class that defines one is refused. The other dunders (operators, comparisons, __len__,
# __iter__, __getitem__, __call__, __hash__, ...) are never called by translated code, since
# every use that would call them (an operator on objects, `==`, `len`, a loop, a subscript, a
# call of an object) is refused where it is written.
REFUSED_DUNDERS = {
    "__new__": "`__new__` (a custom constructor)",
    "__del__": "`__del__` (Python calls it when the object is freed)",
    "__getattr__": "`__getattr__` (computed attributes)",
    "__getattribute__": "`__getattribute__` (computed attributes)",
    "__setattr__": "`__setattr__` (a custom attribute store)",
    "__delattr__": "`__delattr__`",
    "__init_subclass__": "`__init_subclass__`",
    "__class_getitem__": "`__class_getitem__`",
    "__set_name__": "`__set_name__`",
    "__get__": "`__get__` (a descriptor)",
    "__set__": "`__set__` (a descriptor)",
    "__delete__": "`__delete__` (a descriptor)",
    "__format__": "`__format__` (a custom format spec)",
    "__copy__": "`__copy__`",
    "__deepcopy__": "`__deepcopy__`",
}

KEYATTR = re.compile(r"^\[(.*)\]$")  # a string-key subscript rewritten as an attribute: d["k"] -> d.['k']


def keyattr(k):
    return "[%r]" % (k,)


def is_keyattr(a):
    return a.startswith("[") and a.endswith("]")


def key_of_attr(a):
    return ast.literal_eval(a[1:-1])


def literal_number(node):
    """A number, Bool or string literal (a number with a sign): the defaults and class attributes
    Python evaluates once and py2viv may write in place of the name (a string is immutable, so
    writing it again is the same value)."""
    if isinstance(node, ast.Constant):
        return type(node.value) in (int, float, bool, str)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return isinstance(node.operand, ast.Constant) and type(node.operand.value) in (int, float)
    return False


class ClassInfo:
    """One class, named tuple or dict shape, as a Viv record."""

    def __init__(self, name, kind, node):
        self.name = name            # the Python name that constructs it (its constructor's key in funcs)
        self.kind = kind            # "class", "dataclass", "namedtuple", "dict"
        self.node = node
        self.typename = name        # what repr prints (a namedtuple's first argument)
        self.viv = viv_name(name)   # the record's (and the constructor's) Viv name
        self.fields = []            # Python field names, in order (a dict's: its keys' attributes)
        self.fviv = {}              # field -> Viv field name
        self.keys = {}              # dict: field -> the Python key
        self.defaults = {}          # field -> default expression (dataclass, named tuple)
        self.frozen = kind == "namedtuple"
        self.methods = {}           # method name -> func key
        self.props = {}             # property name -> func key
        self.statics = {}           # static (and class) method name -> func key
        self.consts = {}            # class attribute -> (Viv module variable, value node)
        self.fmt = {}               # "__str__"/"__repr__" -> the f-string or string it returns
        self.auto_repr = kind in ("dataclass", "namedtuple", "dict")
        self.refusal = None         # (node, what, hint): the class is refused where it is defined
        self.mutable = False        # some field is rebound after construction (syntactically)
        self.field_types = {}
        self.prev_field_types = {}
        self.field_sites = {}       # field -> a node that set it (for a refusal at the end of a pass)
        self.decl_at = None
        self.dunders = set()
        self.ctor = None            # the constructor's FuncInfo
        self.self_name = "self"

    def refuse(self, node, what, hint):
        if self.refusal is None:
            self.refusal = (node, what, hint)

    def ftype(self, f):
        """The field's type: as this pass has set it, else as the last pass left it."""
        return self.field_types.get(f) or self.prev_field_types.get(f) or UNK

    def words(self):
        if self.kind == "dict":
            return "a dict with the keys %s" % ", ".join(repr(self.keys[f]) for f in self.fields)
        if self.kind == "namedtuple":
            return "a named tuple `%s`" % self.name
        return "an object of class `%s`" % self.name


class ObjectsMixin:
    """The translator's handling of classes and dicts (mixed into py2viv.Translator)."""

    # ================================================================ the module scan

    def _objects_scan(self):
        self.classes = {}           # ctor key (the Python name) -> ClassInfo
        self._class_nodes = {}      # id(ClassDef or Assign) -> ClassInfo
        self.dict_shapes = {}       # the keys, in order -> ClassInfo
        self.method_keys = {}       # method name -> [(func key, takes the object first)]
        self._obj_trees = []        # the constructors' synthesized defs (for the parents map)
        self._dflt = {}             # (def, param) -> (Viv name, T) of a default Python evaluates once
        self._prev_dflt = {}
        self.dc_names, self.dc_mods, self.nt_names, self.nt_mods, self.namedtuple_names, self.coll_mods = \
            set(), set(), set(), set(), set(), set()
        self.dc_replace = set()
        self.typevar_names = set()
        tops = list(self.tree.body)
        for node in self.tree.body:
            if isinstance(node, ast.If) and self._is_main_guard(node):
                tops += node.body
        for node in tops:
            if isinstance(node, ast.ImportFrom):
                for a in node.names:
                    local = a.asname or a.name
                    if node.module == "dataclasses" and a.name == "dataclass":
                        self.dc_names.add(local)
                    elif node.module == "dataclasses" and a.name == "replace":
                        self.dc_replace.add(local)
                    elif node.module == "typing" and a.name == "NamedTuple":
                        self.nt_names.add(local)
                    elif node.module == "typing" and a.name == "TypeVar":
                        self.typevar_names.add(local)
                    elif node.module == "collections" and a.name == "namedtuple":
                        self.namedtuple_names.add(local)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    local = a.asname or a.name
                    if a.name == "dataclasses":
                        self.dc_mods.add(local)
                    elif a.name == "typing":
                        self.nt_mods.add(local)
                    elif a.name == "collections":
                        self.coll_mods.add(local)
        # d["k"] is read as an attribute of the dict (a field of its record): rewritten once here
        self._rewrite_key_subscripts()
        for node in tops:
            if isinstance(node, ast.ClassDef) and id(node) in getattr(self, "enum_nodes", {}):
                continue  # an enum.Enum class (ints.py)
            if isinstance(node, ast.ClassDef):
                info = self._analyze_class(node)
                self._class_nodes[id(node)] = info
                self._register_class(info)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and isinstance(node.value, ast.Call) and self._is_namedtuple_call(node.value):
                info = self._analyze_namedtuple_call(node.targets[0].id, node.value, node)
                self._class_nodes[id(node)] = info
                self._register_class(info)
        self._scan_dict_shapes()
        self._scan_mutations()
        # the attribute names the program reads or calls (a method named nowhere is never run)
        self._attr_names = {n.attr for n in ast.walk(self.tree) if isinstance(n, ast.Attribute)}

    def _rewrite_key_subscripts(self):
        """`X["k"]` (a string constant key) becomes the attribute `X.['k']`: a dict's field. A
        NumPy array has no string keys, so nothing else is read this way; any other base is
        refused where the attribute is translated."""
        class R(ast.NodeTransformer):
            def visit_Subscript(self, n):
                self.generic_visit(n)
                sl = n.slice
                if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                    return ast.copy_location(ast.Attribute(value=n.value, attr=keyattr(sl.value), ctx=n.ctx), n)
                return n
        R().visit(self.tree)
        ast.fix_missing_locations(self.tree)

    def _is_dc_decorator(self, d):
        """The keywords of a @dataclass decorator (ast values), or None if d is not one."""
        f, kws = d, []
        if isinstance(d, ast.Call):
            f, kws = d.func, d.keywords
            if d.args:
                return None
        if isinstance(f, ast.Name) and f.id in self.dc_names or \
                isinstance(f, ast.Attribute) and f.attr == "dataclass" and isinstance(f.value, ast.Name) and f.value.id in self.dc_mods:
            return {k.arg: k.value for k in kws}
        return None

    def _is_namedtuple_base(self, b):
        return isinstance(b, ast.Name) and b.id in self.nt_names or \
            isinstance(b, ast.Attribute) and b.attr == "NamedTuple" and isinstance(b.value, ast.Name) and b.value.id in self.nt_mods

    def _is_namedtuple_call(self, c):
        f = c.func
        return isinstance(f, ast.Name) and f.id in self.namedtuple_names or \
            isinstance(f, ast.Attribute) and f.attr == "namedtuple" and isinstance(f.value, ast.Name) and f.value.id in self.coll_mods

    # ---------------------------------------------------------------- one class
    def _analyze_class(self, c):
        info = ClassInfo(c.name, "class", c)
        for kw in c.keywords:
            info.refuse(kw, "a class with `%s=`" % kw.arg, "Write a plain class, a @dataclass or a NamedTuple.")
        nt = False
        for b in c.bases:
            if isinstance(b, ast.Name) and b.id == "object":
                continue
            if self._is_namedtuple_base(b):
                nt = True
                continue
            if isinstance(b, ast.Subscript) and isinstance(b.value, (ast.Name, ast.Attribute)) \
                    and ast.unparse(b.value) in ("Generic", "typing.Generic"):
                continue
            info.refuse(b, "a class that inherits from `%s`" % ast.unparse(b),
                        "Viv records have no inheritance: put the fields and methods in one class, or write functions.")
        dc = None
        for d in c.decorator_list:
            kws = self._is_dc_decorator(d)
            if kws is None:
                info.refuse(d, "the class decorator `@%s`" % ast.unparse(d), "Only @dataclass is translated.")
                continue
            dc = kws
        # type parameters (`class Stack[T]:`, `Generic[T]`) are annotations only: nothing at run
        # time depends on them, so they are ignored
        if nt:
            info.kind, info.frozen, info.auto_repr = "namedtuple", True, True
            if dc is not None:
                info.refuse(c, "a NamedTuple with @dataclass", "Use one or the other.")
        elif dc is not None:
            info.kind, info.auto_repr = "dataclass", True
            for k, v in dc.items():
                val = v.value if isinstance(v, ast.Constant) else None
                if not isinstance(v, ast.Constant):
                    info.refuse(v, "@dataclass(%s=...) with a computed value" % k, "Write True or False.")
                elif k == "frozen":
                    info.frozen = bool(val)
                elif k == "repr":
                    info.auto_repr = bool(val)
                elif k in ("eq", "match_args", "slots"):
                    pass
                elif (k, val) in (("init", True), ("order", False), ("unsafe_hash", False), ("kw_only", False)):
                    pass
                else:
                    info.refuse(v, "@dataclass(%s=%r)" % (k, val), "Only frozen=, repr=, eq=, slots= and match_args= are translated.")
        body = list(c.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body = body[1:]
        methods = []
        seen_default = None
        for st in body:
            if isinstance(st, ast.Pass) or isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
                continue
            if isinstance(st, ast.FunctionDef):
                methods.append(st)
                continue
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                name = st.target.id
                classvar = "ClassVar" in ast.unparse(st.annotation)
                if info.kind in ("dataclass", "namedtuple") and not classvar:
                    if st.value is not None and isinstance(st.value, ast.Call) and "field" in ast.unparse(st.value.func):
                        info.refuse(st.value, "a dataclass field(...)", "Give the field a number default, or none.")
                        continue
                    if name.startswith("_") and info.kind == "namedtuple":
                        info.refuse(st, "a named tuple field starting with `_`", "Python raises ValueError; rename it.")
                    info.fields.append(name)
                    info.field_sites[name] = st
                    if st.value is not None:
                        if not literal_number(st.value):
                            info.refuse(st.value, "a field default that is not a number or string literal",
                                        "Python evaluates it once for every object (a list or an array default raises "
                                        "ValueError in a dataclass); pass the value to the constructor instead.")
                        info.defaults[name] = st.value
                        seen_default = name
                    elif seen_default is not None:
                        info.refuse(st, "a field without a default after one with a default", "Python raises TypeError here.")
                    continue
                if st.value is None:
                    continue  # an annotation only
                self._class_const(info, name, st.value, st)
                continue
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                if st.targets[0].id == "__slots__":
                    continue
                self._class_const(info, st.targets[0].id, st.value, st)
                continue
            info.refuse(st, "a `%s` statement in a class body" % type(st).__name__,
                        "A class body holds methods, fields (dataclass, NamedTuple) and number constants.")
        init = None
        for m in methods:
            if m.name == "__init__":
                init = m
            elif m.name in REFUSED_DUNDERS:
                info.refuse(m, "a class with %s" % REFUSED_DUNDERS[m.name], "Viv records have no such hook; write a function.")
            elif m.name in ("__str__", "__repr__"):
                info.fmt[m.name] = self._fmt_body(m)
                info.dunders.add(m.name)
            elif m.name.startswith("__") and m.name.endswith("__"):
                info.dunders.add(m.name)  # never called by translated code (its uses are refused)
        if info.kind == "class":
            self._plain_fields(info, init, methods)
        else:
            if init is not None:
                info.refuse(init, "a %s with its own `__init__`" % ("NamedTuple" if info.kind == "namedtuple" else "dataclass"),
                            "Let the class generate __init__ from its fields, or write a plain class.")
            if info.kind == "namedtuple" and not info.fields:
                info.refuse(c, "a NamedTuple without fields", "Give it a field.")
            init = self._synth_init(info, c)
            post = next((m for m in methods if m.name == "__post_init__"), None)
            if post is not None:
                # __post_init__ runs at the end of the generated __init__, on the same object: its
                # body is appended (its own attributes become fields, printed by no repr)
                if info.kind != "dataclass" or len(post.args.args) != 1 or post.args.vararg or post.args.kwarg:
                    info.refuse(post, "this `__post_init__`", "A dataclass's __post_init__ takes `self` only.")
                else:
                    sn = post.args.args[0].arg

                    class Ren(ast.NodeTransformer):
                        def visit_Name(self, n):
                            if n.id == sn:
                                return ast.copy_location(ast.Name(id="self", ctx=n.ctx), n)
                            return n
                    body = [Ren().visit(copy.deepcopy(b)) for b in post.body
                            if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))]
                    init.body = [b for b in init.body if not isinstance(b, ast.Pass)] + body
                    info.repr_fields = list(info.fields)
                    extra = ClassInfo(info.name, "class", c)
                    fake = ast.copy_location(ast.FunctionDef(name="__init__", args=init.args, body=init.body, decorator_list=[],
                                                             returns=None, type_comment=None), post)
                    self._plain_fields(extra, fake, methods)
                    if extra.refusal:
                        info.refuse(*extra.refusal)
                    for f in extra.fields:
                        if f not in info.fields:
                            info.fields.append(f)
                            info.field_sites[f] = extra.field_sites[f]
        self._methods_of(info, methods)
        info.ctor_node = init
        return info

    def _class_const(self, info, name, value, st):
        if not literal_number(value):
            info.refuse(value, "a class attribute that is not a number literal",
                        "Python shares one value between all the objects; make it a number, or a module variable.")
            return
        info.consts[name] = ("%s__%s" % (info.viv, name), value)

    @staticmethod
    def _fmt_body(m):
        """The f-string (or string) a `__str__`/`__repr__` returns, or None."""
        body = m.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body = body[1:]
        if len(body) == 1 and isinstance(body[0], ast.Return) and isinstance(body[0].value, (ast.JoinedStr, ast.Constant)) \
                and (not isinstance(body[0].value, ast.Constant) or isinstance(body[0].value.value, str)) \
                and len(m.args.args) == 1:
            return (m.args.args[0].arg, body[0].value)
        return None

    def _plain_fields(self, info, init, methods):
        """A plain class's fields: every `self.f` set in __init__, first at the top level of its
        body (before anything reads it)."""
        if init is None:
            return
        a = init.args
        if not a.args:
            info.refuse(init, "`__init__` without `self`", "Python raises TypeError when the class is called.")
            return
        if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs:
            info.refuse(init, "`*args`, `**kwargs` or keyword-only parameters in `__init__`", "List the parameters positionally.")
        info.self_name = sn = a.args[0].arg
        tops = {id(s) for s in init.body}
        parent = {}
        for n in ast.walk(init):
            for ch in ast.iter_child_nodes(n):
                parent[ch] = n
        occ = sorted((n for n in ast.walk(init) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                      and n.value.id == sn), key=lambda n: (n.lineno, n.col_offset))
        seen = set()
        for n in occ:
            f = n.attr
            if f in seen or is_keyattr(f):
                continue
            seen.add(f)
            st = n
            while st is not None and not isinstance(st, ast.stmt):
                st = parent.get(st)
            if f in info.consts or any(m.name == f for m in init.body if isinstance(m, ast.FunctionDef)):
                continue
            is_method = any(m.name == f for m in methods)
            if is_method:
                continue  # `self.m(...)`: refused below, as a use of self
            store = isinstance(n.ctx, ast.Store) and not isinstance(st, ast.AugAssign) and id(st) in tops \
                and not (isinstance(st, ast.AnnAssign) and st.value is None)
            if not store:
                what = "read before it is set" if isinstance(n.ctx, ast.Load) or isinstance(st, ast.AugAssign) else "first set inside a block"
                info.refuse(n, "the attribute `%s` %s in `__init__`" % (f, what),
                            "Set every attribute at the top level of __init__ before reading it, so every object has "
                            "all of them (a Viv record has a fixed set of fields).")
                continue
            info.fields.append(f)
            info.field_sites[f] = st

    def _synth_init(self, info, c):
        """The `__init__` a dataclass or NamedTuple generates: one parameter per field, in order,
        with its default, and `self.f = f`."""
        args = [ast.arg(arg="self")] + [ast.arg(arg=f) for f in info.fields]
        defaults = [info.defaults[f] for f in info.fields if f in info.defaults]
        body = []
        for f in info.fields:
            site = info.field_sites[f]
            st = ast.Assign(targets=[ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr=f, ctx=ast.Store())],
                            value=ast.Name(id=f, ctx=ast.Load()))
            body.append(ast.copy_location(st, site))
        if not body:
            body = [ast.copy_location(ast.Pass(), c)]
        fn = ast.FunctionDef(name="__init__", args=ast.arguments(posonlyargs=[], args=args, vararg=None, kwonlyargs=[],
                                                                  kw_defaults=[], kwarg=None, defaults=defaults),
                             body=body, decorator_list=[], returns=None, type_comment=None)
        if hasattr(ast, "TypeAlias"):
            fn.type_params = []
        ast.copy_location(fn, c)
        ast.fix_missing_locations(fn)
        info.self_name = "self"
        return fn

    def _methods_of(self, info, methods):
        for m in methods:
            if m.name == "__init__" or m.name.startswith("__") and m.name.endswith("__"):
                continue
            kind = "method"
            for d in m.decorator_list:
                dn = ast.unparse(d)
                if dn == "property":
                    kind = "property"
                elif dn == "staticmethod":
                    kind = "static"
                elif dn == "classmethod":
                    kind = "classmethod"
                else:
                    info.refuse(d, "the method decorator `@%s`" % dn, "Only @property, @staticmethod and @classmethod are translated.")
            a = m.args
            if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs:
                info.refuse(m, "`*args`, `**kwargs` or keyword-only parameters in the method `%s`" % m.name,
                            "List the parameters positionally.")
            if kind != "static" and not a.args:
                info.refuse(m, "the method `%s` without `self`" % m.name, "Python passes the object first; add `self`.")
            if kind == "property" and len(a.args) != 1:
                info.refuse(m, "a property with parameters", "A property takes `self` only.")
            if m.name in info.fields or m.name in info.consts:
                info.refuse(m, "the method `%s` with the name of an attribute" % m.name, "Rename one of them.")
            for d in (a.defaults or []):
                if not self._const_default(d):
                    info.refuse(d, "a method default that is not a number literal",
                                "Python evaluates it once, when the class is defined; pass the value instead.")
            if kind == "method":
                info.methods[m.name] = m
            elif kind == "property":
                info.props[m.name] = m
            else:
                info.statics[m.name] = (m, kind)

    def _analyze_namedtuple_call(self, name, call, node):
        """`P = namedtuple("P", ["x", "y"])` (or "x y", "x, y"), with `defaults=`."""
        info = ClassInfo(name, "namedtuple", node)
        if len(call.args) != 2 or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
            info.refuse(call, "this namedtuple call", "Write namedtuple(\"Name\", [\"a\", \"b\"]) with literal names.")
            return info
        info.typename = call.args[0].value
        spec = call.args[1]
        names = None
        if isinstance(spec, ast.Constant) and isinstance(spec.value, str):
            names = spec.value.replace(",", " ").split()
        elif isinstance(spec, (ast.List, ast.Tuple)) and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in spec.elts):
            names = [e.value for e in spec.elts]
        if not names or not all(x.isidentifier() and not keyword.iskeyword(x) and not x.startswith("_") for x in names) \
                or len(set(names)) != len(names):
            info.refuse(spec, "these namedtuple field names", "Python raises ValueError, or the names are not literal.")
            return info
        info.fields = names
        for f in names:
            info.field_sites[f] = call
        for kw in call.keywords:
            if kw.arg == "defaults" and isinstance(kw.value, (ast.List, ast.Tuple)) and all(literal_number(e) for e in kw.value.elts):
                ds = kw.value.elts
                if len(ds) > len(names):
                    info.refuse(kw.value, "more defaults than fields", "Python raises TypeError.")
                for f, d in zip(names[len(names) - len(ds):], ds):
                    info.defaults[f] = d
            elif kw.arg == "module":
                pass
            else:
                info.refuse(kw.value, "namedtuple's `%s=`" % kw.arg, "Only defaults= of number literals is translated.")
        info.ctor_node = self._synth_init(info, call)
        return info

    # ---------------------------------------------------------------- registering the defs
    def _register_class(self, info):
        c = info.node
        if info.name in self.funcs:
            info.refuse(c, "a class named like the def `%s`" % info.name, "Rename one of them.")
            return
        for n in ast.walk(self.tree):
            tg = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, (ast.AugAssign, ast.AnnAssign, ast.For)) else []
            if n is c or n is info.node:
                continue
            if any(isinstance(x, ast.Name) and x.id == info.name and isinstance(x.ctx, ast.Store) for t in tg for x in ast.walk(t)):
                info.refuse(n, "the class name `%s` rebound" % info.name, "Use another name for the value.")
        init = info.ctor_node
        if init is None:
            # a plain class without __init__: no fields (Viv records have one at least)
            init = ast.copy_location(ast.FunctionDef(
                name="__init__", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="self")], vararg=None, kwonlyargs=[],
                                                    kw_defaults=[], kwarg=None, defaults=[]),
                body=[ast.copy_location(ast.Pass(), c)], decorator_list=[], returns=None, type_comment=None), c)
            if hasattr(ast, "TypeAlias"):
                init.type_params = []
            ast.fix_missing_locations(init)
        sn = init.args.args[0].arg if init.args.args else "self"
        info.self_name = sn
        for d in init.args.defaults:
            if not self._const_default(d) and info.kind == "class":
                info.refuse(d, "an `__init__` default that is not a number literal",
                            "Python evaluates it once, when the class is defined; pass the value instead.")
        if any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(init)):
            info.refuse(init, "`return` with a value in `__init__`", "Python raises TypeError.")
        elif any(isinstance(n, ast.Return) for n in ast.walk(init)):
            info.refuse(next(n for n in ast.walk(init) if isinstance(n, ast.Return)), "`return` inside `__init__`",
                        "Set every attribute and let __init__ end.")
        # __init__'s body with `self.f` as the local `self__f`
        fields = set(info.fields)
        loc = {f: "self__%s" % f for f in info.fields}
        for f, v in loc.items():
            if v in self.all_names:
                info.refuse(c, "the name `%s` (py2viv uses it for the attribute `%s`)" % (v, f), "Rename it.")
        consts = info.consts

        class Rw(ast.NodeTransformer):
            def visit_Attribute(self, n):
                if isinstance(n.value, ast.Name) and n.value.id == sn:
                    if n.attr in fields:
                        return ast.copy_location(ast.Name(id=loc[n.attr], ctx=n.ctx), n)
                    if n.attr in consts and isinstance(n.ctx, ast.Load):
                        return ast.copy_location(ast.Name(id=consts[n.attr][0], ctx=n.ctx), n)
                self.generic_visit(n)
                return n
        body = [Rw().visit(copy.deepcopy(s)) for s in init.body]
        for n in (m for s in body for m in ast.walk(s)):
            if isinstance(n, ast.Name) and n.id == sn:
                info.refuse(n, "`%s` used inside `__init__` other than to set or read an attribute" % sn,
                            "The object is built when __init__ ends: set its attributes from local values, and call "
                            "its methods after it is made.")
                break
        args = [x for x in init.args.args[1:]]
        ctor = ast.FunctionDef(name=info.name, args=ast.arguments(posonlyargs=[], args=args, vararg=None, kwonlyargs=[],
                                                                   kw_defaults=[], kwarg=None, defaults=list(init.args.defaults)),
                               body=body, decorator_list=[], returns=None, type_comment=None)
        if hasattr(ast, "TypeAlias"):
            ctor.type_params = []
        ast.copy_location(ctor, init)
        ast.fix_missing_locations(ctor)
        f = FuncInfo(ctor)
        f.locals, f.globals_decl = assigned_names(ctor.body)
        f.locals |= set(f.params) | set(loc.values())
        f.is_test = False
        f.ctor_of = info
        f.result_names = list(loc.values())
        info.ctor = f
        info.loc = loc
        self.funcs[info.name] = f
        self._obj_trees.append(ctor)
        # methods, properties, static and class methods
        for kind, table in (("method", info.methods), ("property", info.props)):
            for mname, m in list(table.items()):
                key = self._method_key(info, mname)
                if key is None:
                    continue
                g = FuncInfo(m)
                g.name = key
                g.locals, g.globals_decl = assigned_names(m.body)
                g.locals |= set(g.params)
                g.is_test = False
                g.method_of = info
                g.self_name = g.params[0] if g.params else "self"
                self.funcs[key] = g
                table[mname] = key
                self.method_keys.setdefault(mname, []).append((key, True))
        for mname, (m, kind) in list(info.statics.items()):
            key = self._method_key(info, mname)
            if key is None:
                continue
            node = m
            if kind == "classmethod":
                cls = m.args.args[0].arg
                name = info.name

                class Cls(ast.NodeTransformer):
                    def visit_Name(self, n):
                        if n.id == cls:
                            return ast.copy_location(ast.Name(id=name, ctx=n.ctx), n)
                        return n
                node = Cls().visit(copy.deepcopy(m))
                node.args.args = node.args.args[1:]
                ast.fix_missing_locations(node)
            g = FuncInfo(node)
            g.name = key
            g.locals, g.globals_decl = assigned_names(node.body)
            g.locals |= set(g.params)
            g.is_test = False
            g.static_of = info
            self.funcs[key] = g
            info.statics[mname] = key
            self.method_keys.setdefault(mname, []).append((key, False))
        self.classes[info.name] = info

    def _method_key(self, info, mname):
        key = "%s__%s" % (info.viv, mname)
        if key in self.funcs or key in self.all_names:
            info.refuse(info.node, "the name `%s` (py2viv uses it for the method `%s.%s`)" % (key, info.name, mname), "Rename it.")
            return None
        return key

    # ---------------------------------------------------------------- dict shapes
    def _scan_dict_shapes(self):
        sites = []
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Dict) and n.keys and all(k is not None for k in n.keys):
                ks = [self._dict_key(k) for k in n.keys]
                if all(k is not None for k in ks) and len(set(map(repr, ks))) == len(ks):
                    sites.append((n.lineno, n.col_offset, tuple(ks), n))
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "dict" and not n.args \
                    and n.keywords and all(k.arg is not None for k in n.keywords):
                sites.append((n.lineno, n.col_offset, tuple(k.arg for k in n.keywords), n))
        for _, _, keys, n in sorted(sites, key=lambda s: (s[0], s[1])):
            if keys in self.dict_shapes:
                continue
            k = len(self.dict_shapes) + 1
            while "pydict%d" % k in self.all_names or "pydict%d" % k in self.classes:
                k += 1
            info = ClassInfo("pydict%d" % k, "dict", n)
            for i, k in enumerate(keys):
                f = keyattr(k)
                info.fields.append(f)
                info.keys[f] = k
                info.field_sites[f] = n
                if isinstance(k, str) and k.isidentifier() and not keyword.iskeyword(k):
                    info.fviv[f] = viv_name(k)
                else:
                    info.fviv[f] = ("k%d" % k if k >= 0 else "kn%d" % -k) if isinstance(k, int) else "k_%d" % i
            if len(set(info.fviv.values())) != len(info.fviv):
                info.fviv = {f: "k_%d" % i for i, f in enumerate(info.fields)}
            self.dict_shapes[keys] = info

    @staticmethod
    def _dict_key(k):
        if isinstance(k, ast.Constant) and (isinstance(k.value, str) or type(k.value) is int):
            return k.value
        if isinstance(k, ast.UnaryOp) and isinstance(k.op, ast.USub) and isinstance(k.operand, ast.Constant) \
                and type(k.operand.value) is int:
            return -k.operand.value
        return None

    def _scan_mutations(self):
        """A class (or dict shape) is mutable when some field is rebound after construction:
        an attribute store `x.f = v`, `x.f += v` outside the constructors, with f one of its
        fields (by name: the scan does not know x's class, so every class with such a field
        counts)."""
        by_field = {}
        for info in list(self.classes.values()) + list(self.dict_shapes.values()):
            for f in info.fields:
                by_field.setdefault(f, []).append(info)
        # a plain class's __init__ sets its fields (it builds the object: not a rebinding)
        inits = {id(n) for info in self.classes.values() if isinstance(info.node, ast.ClassDef)
                 for m in info.node.body if isinstance(m, ast.FunctionDef) and m.name == "__init__" for n in ast.walk(m)}
        for n in ast.walk(self.tree):
            if id(n) in inits:
                continue
            tgts = []
            if isinstance(n, ast.Assign):
                tgts = n.targets
            elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
                tgts = [n.target]
            for t in tgts:
                for x in ast.walk(t):
                    if isinstance(x, ast.Attribute) and isinstance(x.ctx, ast.Store):
                        for info in by_field.get(x.attr, []):
                            info.mutable = True
                    elif isinstance(x, ast.Subscript) and isinstance(x.ctx, ast.Store) and self._dict_key(x.slice) is not None:
                        # d[1] = v: every dict shape with that int key (an array store looks the same)
                        for info in by_field.get(keyattr(self._dict_key(x.slice)), []):
                            info.mutable = True

    # ================================================================ each pass

    def _objects_pass_start(self):
        for info in self._all_infos():
            info.prev_field_types = info.field_types
            info.field_types = {}
            info.decl_at = None
            info.used = False
        for f in self.funcs.values():
            f.prev_threaded = getattr(f, "prev_threaded", set()) | getattr(f, "threaded", set())
            f.threaded = set()
            if getattr(f, "vret_set", False):
                f.prev_vret = (f.vret,)
            elif not hasattr(f, "prev_vret"):
                f.prev_vret = None
            f.vret = None
            f.vret_set = False
            ci = getattr(f, "ctor_of", None)
            if ci is not None:
                f.ret = T("rec", name=ci.name)
            mi = getattr(f, "method_of", None)
            if mi is not None and f.params:
                f.param_types[0] = T("rec", name=mi.name)
        self._prev_dflt = {**self._prev_dflt, **self._dflt}
        self._dflt = {}

    def _all_infos(self):
        return list(self.classes.values()) + list(self.dict_shapes.values())

    def _objects_state(self):
        out = []
        for info in self._all_infos():
            out.append((info.name, tuple(sorted((k, v) for k, v in info.field_types.items()))))
        for k, f in sorted(self.funcs.items()):
            th = getattr(f, "prev_threaded", set()) | getattr(f, "threaded", set())
            if th:
                out.append((k, tuple(sorted(th)), getattr(f, "vret", None), getattr(f, "vret_set", False)))
        return tuple(out)

    @staticmethod
    def _default_text(node):
        """A dataclass or named-tuple default (a literal, `literal_number`) as a Viv record field
        default: the same constant, so a literal that leaves the field out
        gets what Python's constructor gives. None when it has no exact spelling here."""
        neg = isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
        v = node.operand.value if isinstance(node, ast.UnaryOp) else node.value
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return ("-" if neg else "") + str(v)
        if isinstance(v, float):
            r = repr(v)
            if r in ("inf", "nan") or ("." not in r and "e" not in r):
                return None
            return ("-" if neg else "") + r
        if isinstance(v, str) and all(c.isascii() and c.isprintable() and c not in '"\\{}' for c in v):
            return '"%s"' % v
        return None

    def _record_decl(self, info):
        parts = []
        for f in info.fields or []:
            part = "%s: %s" % (self._fv(info, f), self._viv_type(info.ftype(f), info, f))
            d = info.defaults.get(f)
            text = self._default_text(d) if d is not None else None
            if text is not None:
                part += " = " + text
            parts.append(part)
        if not parts:
            parts = ["py_unit: Num"]
        return "record %s { %s }" % (info.viv, ", ".join(parts))

    def _objects_finish(self):
        """At the end of a pass: the records' declarations, with their fields' types, at the top
        of the program (Viv declares records at module scope; a record's fields' records come
        first)."""
        used = [info for info in self.classes.values() if info.decl_at is not None] + \
            [info for info in self.dict_shapes.values() if info.used]
        order, seen = [], set()

        def visit(info, path):
            if info.name in seen or info.name in path:
                return
            path = path | {info.name}
            for f in info.fields:
                ti = self._rec_info(info.ftype(f))
                if ti is not None:
                    visit(ti, path)
            seen.add(info.name)
            order.append(info)
        for info in used:
            visit(info, frozenset())
        self.out[0:0] = [(0, self._record_decl(info), getattr(info.node, "lineno", 0), node_pos(info.node)) for info in order]
        # a class whose objects hold objects of their own class, directly or not
        for info in self.classes.values():
            if info.decl_at is None:
                continue
            seen, stack = set(), [info.name]
            while stack:
                x = stack.pop()
                xi = self._info_named(x)
                if xi is None:
                    continue
                for f in xi.fields:
                    t = xi.ftype(f)
                    if t.kind == "rec":
                        if t.name == info.name:
                            self.refusals.append(Refusal(info.node, "a class whose objects hold an object of the class `%s` "
                                                         "(a linked structure)" % info.name,
                                                         "Viv records hold numbers, arrays and other records by value; "
                                                         "keep the links as indices into arrays."))
                            stack = []
                            break
                        if t.name not in seen:
                            seen.add(t.name)
                            stack.append(t.name)

    def _viv_type(self, t, info, f):
        if t.kind == "num":
            return "Num"
        if t.kind == "bool":
            return "Bool"
        if t.kind == "arr":
            return "Arr"
        if t.kind == "carr":
            return "CArr"
        if t.kind == "rec":
            return self._info_named(t.name).viv
        if t.kind == "str":
            return "Str"
        if t.kind == "strs":
            return "[Str]"
        return "Num"  # unknown in this pass: a refusal elsewhere, or known on the next pass

    def _info_named(self, name):
        if name in self.classes:
            return self.classes[name]
        for info in self.dict_shapes.values():
            if info.name == name:
                return info
        return None

    def _rec_info(self, t):
        return self._info_named(t.name) if t is not None and t.kind == "rec" else None

    def _field_set(self, info, f, t, node):
        """Record that field f of info holds a value of type t (set at node)."""
        if t.kind in ("none", "func", "module", "tuple", "rng", "cnum") or t.kind == "arr" and t.container == "shape":
            what = {"none": "None", "func": "a function", "module": "a module", "tuple": "several values",
                    "rng": "a Generator (np.random.default_rng)", "cnum": "a complex number"}.get(t.kind, "a shape (a tuple)")
            self.refuse(node, "%s held in the %s `%s` of %s" % (what, "key" if info.kind == "dict" else "attribute",
                                                               self._fdisp(info, f), info.words()),
                        "A Viv record's fields hold numbers, Bools, strings, arrays and other records.")
        if t.kind == "rec":
            ti = self._rec_info(t)
            if ti is not None and ti.mutable:
                self.refuse(node, "%s held in the %s `%s` of %s" % (ti.words(), "key" if info.kind == "dict" else "attribute",
                                                                   self._fdisp(info, f), info.words()),
                            "The field would be a second reference to an object whose fields are rebound elsewhere "
                            "(Python shares it, a Viv record holds a copy); keep the object in a variable.")
        if t.kind == "unknown":
            return
        old = info.ftype(f)
        j = join(info.field_types.get(f), t)
        mixed = old.kind in ("num", "bool") and t.kind in ("num", "bool") and old.kind != t.kind
        if j is None or mixed:
            self.refuse(node, "the %s `%s` of %s holds %s here and %s elsewhere" % (
                "key" if info.kind == "dict" else "attribute", self._fdisp(info, f), info.words(),
                self._kind_words(t), self._kind_words(old)),
                "A Viv record's field keeps one type: use two attributes.")
        info.field_types[f] = j

    @staticmethod
    def _fv(info, f):
        """A field's Viv name."""
        return info.fviv.get(f) or viv_name(f)

    @staticmethod
    def _fdisp(info, f):
        return repr(info.keys[f]) if info.kind == "dict" else f

    # ================================================================ statements

    def _s_ClassDef(self, s, top):
        if not top or self.scope is not None:
            self.refuse(s, "a class inside a function or a block", "Define the class at module level.")
        if id(s) in getattr(self, "enum_nodes", {}):
            self._ie_enum_stmt(s)  # an enum.Enum class: a Viv enum (ints.py)
            return
        info = self._class_nodes.get(id(s))
        if info is None:
            self.refuse(s, "a class", "Not in the numeric subset.")
        self._class_stmt(info, s)

    def _objects_assign(self, tgt, value, s):
        """`P = namedtuple(...)` at module level: the named tuple's record and constructor.
        `T = TypeVar("T")`: a type variable for annotations and `Generic[T]`, which nothing at
        run time reads: no Viv."""
        if isinstance(value, ast.Call) and self.scope is None and isinstance(tgt, ast.Name) \
                and (isinstance(value.func, ast.Name) and value.func.id in self.typevar_names or
                     isinstance(value.func, ast.Attribute) and value.func.attr == "TypeVar"
                     and isinstance(value.func.value, ast.Name) and value.func.value.id in self.nt_mods):
            return True
        info = self._class_nodes.get(id(s))
        if info is None:
            return False
        if self.scope is not None:
            self.refuse(s, "a namedtuple made inside a function", "Make it at module level.")
        self._class_stmt(info, s)
        return True

    def _class_stmt(self, info, s):
        if info.refusal:
            node, what, hint = info.refusal
            self.refuse(node, what, hint)
        self._flush_comments(s.lineno)
        info.decl_at = len(self.out)  # the record is declared at the top (_objects_finish)
        for name, (vn, value) in info.consts.items():
            code, t, _ = self.expr(value)
            self.set_var(vn, t, s, may_version=False)
            self.emit("%s = %s" % (vn, code), s)
        self._obj_def(info.ctor, info.ctor.node, ctor=info)
        for table in (info.methods, info.props, info.statics):
            for mname, key in table.items():
                if mname not in self._attr_names:
                    continue  # never called: Python never runs it, and its parameters have no types
                f = self.funcs[key]
                self._obj_def(f, f.node)

    def _obj_def(self, f, node, ctor=None):
        """A constructor, method, property or static method as a Viv def."""
        outer_types = self.types
        saved_indent = self.indent
        self.scope = f
        try:
            body = node.body
            if ctor is None and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                for dl in body[0].value.value.strip().splitlines():
                    self.emit("#| " + dl.strip() if dl.strip() else "#|", None)
                body = body[1:]
            params = []
            for i, p in enumerate(f.params):
                pt = self._param_type(f, i)
                self.set_var(p, pt, node)
                params.append(self.vname(p) + self._annot(pt))
            # a constructor def returns its record
            ret = " -> %s" % ctor.viv if ctor is not None else ""
            self.emit("def %s(%s)%s {" % (viv_name(f.name), ", ".join(params), ret), node)
            self.indent += 1
            self._block(body)
            if ctor is not None:
                vals = []
                for fld in ctor.fields:
                    ln = ctor.loc[fld]
                    t = self.var_type(ln)
                    if t is None:
                        r = Refusal(node, "the attribute `%s` is not set (its assignment was refused)" % fld, "Fix that refusal.")
                        r.cascade = True
                        raise r
                    self._field_set(ctor, fld, t, ctor.field_sites.get(fld, node))
                    vals.append("%s: %s" % (self._fv(ctor, fld), self.vname(ln)))
                if not vals:
                    vals = ["py_unit: 0"]
                self.emit("return %s { %s }" % (ctor.viv, ", ".join(vals)), node)
            else:
                self._obj_def_tail(f, node)
            self.indent -= 1
            self.emit("}", node)
        except Refusal as r:
            self.refusals.append(r)
            self.out.append((self.indent, "# REFUSED: " + r.what, getattr(node, "lineno", 0), node_pos(node)))
        finally:
            self.indent = saved_indent
            self.scope = None
            self.types = outer_types

    # ---------------------------------------------------------------- state threading
    def _threaded(self, f):
        f = getattr(f, "base", f)  # a clone (infer.py) threads as its def does
        return sorted(getattr(f, "prev_threaded", set()) | getattr(f, "threaded", set()))

    def _obj_def_tail(self, f, node):
        """The end of a def that rebinds a field of an object passed to it: it returns the
        object (the objects), and its callers rebind their variables."""
        thr = self._threaded(f)
        if not thr:
            return
        for k in thr:
            p = f.params[k]
            if self._assigns_name(f.node.body, p):
                self.refuse(node, "`%s` rebound inside `%s`, which also changes the object passed as `%s`" % (p, self._fname(f), p),
                            "The caller's object would not see the rebinding in Python; use another name.")
        body = f.node.body
        last = body[-1] if body else None
        if isinstance(last, (ast.Return, ast.Raise)):
            return
        vret = self._vret(f)
        if vret is not None:
            self.refuse(node, "`%s` changes `%s` and returns a value on some paths only" % (self._fname(f), f.params[thr[0]]),
                        "Return a value on every path.")
        self._set_thread_ret(f, [])
        self.emit("return %s" % ", ".join(self.vname(f.params[k]) for k in thr), node)

    @staticmethod
    def _vret(f):
        """The values a threaded def returns besides the objects (None for none, a T, or a
        tuple T), from this pass or the last."""
        if getattr(f, "vret_set", False):
            return f.vret
        pv = getattr(f, "prev_vret", None)
        return pv[0] if pv else None

    @staticmethod
    def _fname(f):
        mi = getattr(f, "method_of", None) or getattr(f, "static_of", None)
        if mi is not None:
            return "%s.%s" % (mi.name, f.node.name)
        return f.name

    def _set_thread_ret(self, f, vals):
        """The Viv return of a threaded def: the objects, then its own values."""
        thr = self._threaded(f)
        vt = None if not vals else (vals[0] if len(vals) == 1 else tup(vals))
        if f.vret_set:
            old = f.vret
            if (old is None) != (vt is None) or old is not None and join(old, vt) is None:
                return False
            f.vret = join(old, vt) if old is not None else None
        else:
            f.vret, f.vret_set = vt, True
        types = [self._param_type(f, k) for k in thr]
        vals_t = [] if f.vret is None else (list(f.vret.elts) if f.vret.kind == "tuple" else [f.vret])
        allt = types + vals_t
        f.ret = allt[0] if len(allt) == 1 else tup(allt)
        return True

    def _obj_return(self, s):
        """`return ...` in a def that rebinds a field of an object passed to it."""
        f = self.scope
        if f is None or getattr(f, "ctor_of", None) is not None:
            return False
        thr = self._threaded(f)
        v = s.value
        if not thr:
            if v is not None and self._threaded_call_of(v) is not None:
                # `return obj.m()`: the call on its own line, then the return of its value
                tmp = ast.copy_location(ast.Name(id=self.new_tmp("r"), ctx=ast.Store()), s)
                self._obj_thread_call(v, s, [tmp])
                self._s_Return(ast.copy_location(ast.Return(value=ast.copy_location(ast.Name(id=tmp.id, ctx=ast.Load()), s)), s), False)
                return True
            return False
        codes, types = [], []
        if v is None or isinstance(v, ast.Constant) and v.value is None:
            pass
        elif isinstance(v, ast.Tuple):
            for e in v.elts:
                if isinstance(e, (ast.Starred, ast.Tuple)):
                    self.refuse(e, "a starred or nested value in a return", "Return the values flat.")
                c, t, _ = self.expr(e)
                codes.append(c)
                types.append(t)
        else:
            if self._threaded_call_of(v) is not None:
                tmp = ast.copy_location(ast.Name(id=self.new_tmp("r"), ctx=ast.Store()), s)
                self._obj_thread_call(v, s, [tmp])
                v = ast.copy_location(ast.Name(id=tmp.id, ctx=ast.Load()), s)
            c, t, _ = self.expr(v)
            if t.kind == "tuple":
                self.refuse(v, "returning several values of a call from a def that changes `%s`" % f.params[thr[0]],
                            "Unpack them first: `a, b = g(...)`, then `return a, b`.")
            codes.append(c)
            types.append(t)
        for c, t, e in zip(codes, types, (v.elts if isinstance(v, ast.Tuple) else [v])):
            if t.kind in ("none", "func", "module"):
                self.refuse(e, "returning %s" % self._kind_words(t), "Return numbers, bools, strings, arrays and records.")
        if not self._set_thread_ret(f, types):
            self.refuse(s, "`%s` returns %s here and %s elsewhere" % (self._fname(f), "a value" if types else "nothing",
                                                                     "nothing" if types else "a value"),
                        "A def that changes an object passed to it returns the same number of values from every `return`.")
        self.flush_pre(s)
        self.emit("return %s" % ", ".join([self.vname(f.params[k]) for k in thr] + codes), s)
        return True

    def _threaded_call_of(self, v):
        """(func key, the call with the object first) when v is a call of a def or method that
        changes an object passed to it; else None."""
        if not isinstance(v, ast.Call):
            return None
        if isinstance(v.func, ast.Name) and v.func.id in self.funcs:
            f = self.funcs[v.func.id]
            return (v.func.id, v) if self._threaded(f) else None
        if isinstance(v.func, ast.Attribute):
            key, synth = self._method_target(v)
            if key is not None and self._threaded(self.funcs[key]):
                return key, synth
        return None

    def _method_target(self, call):
        """(func key, the call as a call of that key with the receiver first) for a method call
        on an object whose class is known here; (None, None) otherwise."""
        f = call.func
        t = self._static_type(f.value)
        info = self._rec_info(t)
        if info is not None:
            key = info.methods.get(f.attr)
            if key is not None:
                synth = ast.copy_location(ast.Call(func=ast.copy_location(ast.Name(id=key, ctx=ast.Load()), f),
                                                   args=[f.value] + list(call.args), keywords=list(call.keywords)), call)
                return key, synth
        return None, None

    def _static_type(self, node):
        """The type of a name or a chain of attributes of one, without translating anything."""
        if isinstance(node, ast.Name):
            return self.var_type(node.id)
        if isinstance(node, ast.Attribute):
            bt = self._static_type(node.value)
            info = self._rec_info(bt)
            if info is not None and node.attr in info.fields:
                return info.ftype(node.attr)
        return None

    def _obj_thread_stmt(self, value, s, targets):
        """A statement `obj.m(...)`, `x = obj.m(...)`, `a, b = f(obj, ...)` whose call changes an
        object passed to it: the call rebinds the caller's variables. True when handled."""
        if self._threaded_call_of(value) is None:
            return False
        self._obj_thread_call(value, s, targets)
        return True

    def _obj_thread_call(self, value, s, targets):
        key, synth = self._threaded_call_of(value)
        f = self.funcs[key]
        thr = self._threaded(f)
        bound = {}
        for i, a in enumerate(synth.args):
            if isinstance(a, ast.Starred):
                self.refuse(a, "a starred argument", "Pass the values one by one.")
            if i < len(f.params):
                bound[i] = a
        for kw in synth.keywords:
            if kw.arg in f.params:
                bound[f.params.index(kw.arg)] = kw.value
        names = []
        for k in thr:
            a = bound.get(k)
            p = f.params[k]
            if a is None:
                self.refuse(value, "a call of `%s` that changes its default for `%s`" % (self._fname(f), p),
                            "Python changes the one default object every call shares; pass the object.")
            if not isinstance(a, ast.Name):
                self.refuse(a, "a call of `%s`, which changes the object passed as `%s`, with an expression there" % (self._fname(f), p),
                            "Pass a variable: the call gives the changed object back to it.")
            names.append(a.id)
        if len(set(names)) != len(names):
            self.refuse(value, "the same object passed twice to `%s`, which changes it" % self._fname(f),
                        "Python changes one object through both parameters, Viv two copies; pass it once.")
        for i, a in enumerate(synth.args):
            for x in ast.walk(a):
                if isinstance(x, ast.Name) and x.id in names and x is not bound.get(thr[names.index(x.id)]) \
                        and not isinstance(self.parents.get(x), (ast.Attribute, ast.Subscript)):
                    self.refuse(x, "`%s` passed again to `%s`, which changes it" % (x.id, self._fname(f)),
                                "Python sees the changes through both, Viv through one; pass it once.")
        for nm in names:
            self._obj_rebind_ok(nm, value)
            if self.scope is None and nm in self.global_reads:
                self.refuse(value, "a call of `%s`, which changes the module's object `%s`, which a def reads as a module variable"
                            % (self._fname(f), nm),
                            "During the call the def would see the object as it was before the call (the translation gives "
                            "the changed object back when the call returns); pass the object to that def as an argument.")
        self.tuple_site = synth
        self._thread_site = synth
        try:
            code, ret, _ = self._user_call(synth)
        finally:
            self.tuple_site = None
            self._thread_site = None
        vret = self._vret(f)
        vals = [] if vret is None else (list(vret.elts) if vret.kind == "tuple" else [vret])
        if targets is None:
            tnodes = []
            for _ in vals:
                tnodes.append(ast.copy_location(ast.Name(id=self.new_tmp("drop"), ctx=ast.Store()), s))
        else:
            tnodes = list(targets)
            if len(tnodes) != len(vals):
                if not vals:
                    self.refuse(value, "the value of `%s`, which returns nothing" % self._fname(f), "Python gives None; call it on its own line.")
                self.refuse(value, "unpacking %d values into %d names" % (len(vals), len(tnodes)), "Give one name per value.")
        objs = []
        for nm, k in zip(names, thr):
            objs.append(ast.copy_location(ast.Name(id=nm, ctx=ast.Store()), s))
        types = [self._param_type(f, k) for k in thr] + vals
        later = []
        for i, tn in enumerate(tnodes):
            if not isinstance(tn, (ast.Name, ast.Subscript)):
                tmp = ast.copy_location(ast.Name(id=self.new_tmp("r"), ctx=ast.Store()), s)
                later.append((tn, tmp.id, vals[i]))
                tnodes[i] = tmp
        self._unpack_into(objs + tnodes, code, types, s)
        for tn, tmp, t in later:
            self._store(tn, self.vname(tmp), t, s)

    def _obj_rebind_ok(self, nm, node):
        """The variable `nm`, which holds an object, may be rebound here: a local or parameter of
        the current def (a parameter makes the def return it too), or a module variable at
        module level."""
        f = self.scope
        if f is not None and (nm not in f.locals or nm in f.globals_decl):
            self.refuse(node, "a change to the module's object `%s` inside `%s`" % (nm, self._fname(f)),
                        "The translation rebinds the variable that holds the object, and a def cannot rebind a module "
                        "variable; pass the object as an argument (the def then gives it back).")
        if f is not None and nm in f.params:
            if getattr(f, "ctor_of", None) is not None:
                self.refuse(node, "`__init__` changes the object passed as `%s`" % nm,
                            "Change it before or after constructing.")
            getattr(f, "base", f).threaded.add(f.params.index(nm))  # a clone's def (infer.py) is then one def

    # ---------------------------------------------------------------- stores
    def _obj_store(self, tgt, code, t, s):
        """`obj.f = v` (a field rebound) and `d["k"] = v`: the variable is rebound to a new record."""
        base = tgt.value
        bt = self._static_type(base) if isinstance(base, (ast.Name, ast.Attribute)) else None
        info = self._rec_info(bt)
        if info is None:
            if is_keyattr(tgt.attr):
                self.refuse(tgt, "a store with the key %s into %s" % (tgt.attr[1:-1], self._kind_words(bt or UNK)),
                            "Only a dict built from a literal with constant keys is translated, and its keys are fixed.")
            return None
        if not isinstance(base, ast.Name):
            self.refuse(tgt, "assigning a field of a field (`a.b.c = v`)",
                        "Viv records are values: build the inner object anew and assign it to `a.b` from a variable.")
        f = tgt.attr
        if f not in info.fields:
            if info.kind == "dict":
                self.refuse(tgt, "adding the key %s to %s" % (f[1:-1], info.words()),
                            "A translated dict keeps the keys of its literal; put every key in the literal.")
            self.refuse(tgt, "a new attribute `%s` of %s (not set in __init__)" % (f, info.words()),
                        "Set every attribute in __init__.")
        if info.kind == "namedtuple":
            self.refuse(tgt, "assigning a field of a named tuple", "Python raises AttributeError; use p._replace(%s=...)." % f)
        if info.frozen:
            self.refuse(tgt, "assigning a field of a frozen dataclass", "Python raises FrozenInstanceError; use dataclasses.replace.")
        self._field_set(info, f, t, s)
        self._obj_rebind_ok(base.id, tgt)
        self._obj_emit_rebind(base.id, info, {f: code}, s)
        return True

    def _obj_emit_rebind(self, nm, info, repl, s):
        v = self.vname(nm)
        fields = ", ".join("%s: %s" % (self._fv(info, f), repl[f] if f in repl else "%s.%s" % (v, self._fv(info, f)))
                           for f in info.fields)
        self.flush_pre(s)
        self.emit("%s = %s { %s }" % (v, info.viv, fields), s)

    def _obj_augassign(self, s):
        """`obj.f += v`: a number field is rebound; an array field is changed in place, as NumPy
        does (and Python then sets the same array back)."""
        tgt = s.target
        if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name):
            # d[1] += v of a dict with int keys: its field
            di = self._rec_info(self._static_type(tgt.value))
            k = self._dict_key(tgt.slice)
            if di is not None and di.kind == "dict" and k is not None:
                tgt = ast.copy_location(ast.Attribute(value=tgt.value, attr=keyattr(k), ctx=ast.Store()), tgt)
        if not isinstance(tgt, ast.Attribute):
            return False
        base = tgt.value
        bt = self._static_type(base) if isinstance(base, (ast.Name, ast.Attribute)) else None
        info = self._rec_info(bt)
        if info is None:
            return False
        f = tgt.attr
        if f not in info.fields:
            self.refuse(tgt, "`%s=` of the attribute `%s`, which %s does not have" % (self._opsym(s.op), f, info.words()),
                        "Python raises AttributeError (KeyError for a dict).")
        if info.kind == "namedtuple":
            self.refuse(tgt, "assigning a field of a named tuple", "Python raises AttributeError.")
        if info.frozen:
            self.refuse(tgt, "assigning a field of a frozen dataclass", "Python raises FrozenInstanceError.")
        ft = info.ftype(f)
        load = ast.copy_location(ast.Attribute(value=base, attr=f, ctx=ast.Load()), tgt)
        full, rt, _ = self.expr(ast.copy_location(ast.BinOp(left=load, op=s.op, right=s.value), s))
        if ft.is_arr():
            if ft.container != "nd":
                self.refuse(s, "`%s=` of a list held in a field" % self._opsym(s.op),
                            "Python changes the list in place (it extends it for `+=`); write the loop.")
            vt = self.expr(s.value)[1]
            if ft.kind == "arr" and vt.kind in ("cnum", "carr") or ft.py == "bool" \
                    or ft.py == "int" and isinstance(s.op, ast.Div) \
                    or ft.py == "int" and vt.kind != "unknown" and (vt.py not in ("int", "bool") or vt.kind in ("cnum", "carr")):
                self.refuse(s, "`%s=` of an array whose dtype cannot hold the result" % self._opsym(s.op),
                            "NumPy raises (it cannot cast the result back in place); make the array float.")
            if rt.is_arr() and rt.rank != ft.rank:
                self.refuse(s, "`%s=` of a rank-%d array with a rank-%d result" % (self._opsym(s.op), ft.rank, rt.rank),
                            "NumPy raises (the broadcast result does not fit the array).")
            fc = self.expr(load)[0]
            if vt.is_arr():
                fit = self.use("py_cinpl" if ft.kind == "carr" else "py_inpl")
            else:
                fit = self.use("py_cfit" if ft.kind == "carr" else "py_fit")
            self.flush_pre(s)
            self.emit("%s[:] = %s(%s, %s)" % (fc, fit, full, fc), s)
            return True
        if not isinstance(base, ast.Name):
            self.refuse(tgt, "assigning a field of a field (`a.b.c += v`)", "Build the inner object anew.")
        self._field_set(info, f, rt, s)
        self._obj_rebind_ok(base.id, tgt)
        self._obj_emit_rebind(base.id, info, {f: full}, s)
        return True

    def _obj_store_base(self, node):
        """The array a store writes into when it is a field: `p.v[i] = x`, `d["k"][i] = x`:
        (Viv code, type), or None."""
        bt = self._static_type(node.value) if isinstance(node, ast.Attribute) else None
        info = self._rec_info(bt)
        if info is None or node.attr not in info.fields:
            return None
        t = info.ftype(node.attr)
        return self.expr(node)[0], t

    # ================================================================ expressions

    def _obj_use(self, node, t):
        """Where an object (a record) or a class may appear. A class is only called or read
        (C(...), C.K, C.m(...)). An object of a class whose fields are rebound somewhere may not
        get a second name: Python would share it, and Viv copies it."""
        p = self.parents.get(node)
        if t.kind == "func":
            if t.name not in self.classes:
                return
            if p is None or isinstance(p, ast.Attribute) and p.value is node or isinstance(p, ast.Call) and p.func is node:
                return
            self.refuse(node, "the class `%s` as a value" % t.name, "Call it, or read its attributes.")
        info = self._rec_info(t)
        if info is None or not info.mutable or p is None:
            return
        if isinstance(p, (ast.Attribute, ast.Subscript)) and p.value is node:
            return
        if isinstance(p, ast.Call) and node in p.args or isinstance(p, ast.keyword):
            return
        if isinstance(p, (ast.FormattedValue, ast.Compare, ast.List)):
            return  # printed, or refused where it is translated (a comparison, a list of objects)
        if isinstance(p, (ast.Assign, ast.AnnAssign)) and p.value is node:
            if isinstance(node, (ast.Call, ast.Dict)):
                return  # a new object
            self.refuse(node, "a second name for %s, whose fields are rebound" % info.words(),
                        "Python shares one object between the names, and Viv copies the record; use one name "
                        "(or make a copy with a new constructor call).")
        if isinstance(p, ast.Return) or isinstance(p, ast.Tuple) and isinstance(self.parents.get(p), ast.Return):
            f = self.scope
            if isinstance(node, ast.Name) and (f is None or node.id in f.params or node.id not in f.locals):
                self.refuse(node, "returning %s `%s` that the caller also holds" % (info.words(), node.id),
                            "Python returns the same object, and Viv a copy; its fields are rebound elsewhere, so the "
                            "two would differ. Return a new object.")
            if isinstance(node, ast.Name) or isinstance(node, ast.Call):
                return
        if isinstance(p, ast.Expr):
            return
        self.refuse(node, "%s used here, whose fields are rebound" % info.words(),
                    "An object whose fields are rebound may be read, passed to a def, and returned when made in the def; "
                    "Python shares it where Viv would copy it here.")

    def _private_check(self, n, a, info):
        """`obj.__x` outside the class: Python mangles `__x` inside the class body only (to
        `_C__x`), so outside it the name is not the attribute's and Python raises."""
        if a.startswith("__") and not a.endswith("__") and info is not None:
            f = self.scope
            mine = f is not None and info in (getattr(f, "method_of", None), getattr(f, "ctor_of", None), getattr(f, "static_of", None))
            if not mine:
                self.refuse(n, "the private attribute `%s` outside the class `%s`" % (a, info.name),
                            "Python mangles `%s` to `_%s%s` inside the class, so this raises AttributeError." % (a, info.name.lstrip("_"), a))

    def _obj_attr(self, n, base, bt, bp):
        """An attribute of an object (`p.x`, `p.area` of a property, `d["k"]`), of a class
        (`C.K`), or a string key of something that is not a dict. None when not ours."""
        a = n.attr
        self._private_check(n, a, self.classes.get(bt.name) if bt.kind == "func" else self._rec_info(bt))
        if bt.kind == "func" and bt.name in self.classes:
            info = self.classes[bt.name]
            if a in info.consts:
                return self._const_read(info, a)
            if a in info.methods or a in info.statics or a in info.props:
                self.refuse(n, "the method `%s.%s` as a value" % (bt.name, a), "Call it.")
            self.refuse(n, "the attribute `%s` of the class `%s`" % (a, bt.name), "Python raises AttributeError.")
        info = self._rec_info(bt)
        if info is None:
            if is_keyattr(a):
                if bt.kind == "unknown":
                    r = Refusal(n, "a subscript with the key %s of a value of unknown type" % a[1:-1], "Fix the refusal that produced it.")
                    r.cascade = True
                    raise r
                self.refuse(n, "a subscript with the key %s of %s" % (a[1:-1], self._kind_words(bt)),
                            "Only a dict built from a literal with constant keys is translated.")
            return None
        if a in info.fields:
            return "%s.%s" % (self._paren(base, bp, 100), info.fviv.get(a) or viv_name(a)), info.ftype(a), 100
        if a in info.props:
            return self._obj_call(info.props[a], n, [n.value], [], (base, bt, bp))
        if a in info.consts:
            if not isinstance(n.value, ast.Name):
                self.refuse(n, "a class attribute read through an expression", "Read it through the class: `%s.%s`." % (info.name, a))
            return self._const_read(info, a)
        if info.kind == "dict":
            self.refuse(n, "the key %s, which %s does not have" % (a[1:-1], info.words()), "Python raises KeyError.")
        if a in info.methods or a in info.statics:
            self.refuse(n, "the method `%s` as a value" % a, "Call it: `obj.%s(...)`." % a)
        self.refuse(n, "the attribute `%s`, which %s does not have" % (a, info.words()), "Python raises AttributeError.")

    def _const_read(self, info, a):
        vn = info.consts[a][0]
        t = self.var_type(vn)
        if t is None:
            t = UNK
        return vn, t, 100

    def _obj_call(self, key, node, args, keywords, recv=None):
        """A call of the def `key` (a method, property or static method) with these argument nodes."""
        synth = ast.copy_location(ast.Call(func=ast.copy_location(ast.Name(id=key, ctx=ast.Load()), node),
                                           args=list(args), keywords=list(keywords)), node)
        if recv is not None:
            self._xmemo[id(args[0])] = recv  # the receiver, translated once already
        return self._user_call(synth)

    def _obj_method(self, n, base, bt, bp):
        """`obj.m(...)`, `C.static(...)`, `p._replace(...)`, `d.get(...)`, ... None when not ours."""
        m = n.func.attr
        self._private_check(n, m, self.classes.get(bt.name) if bt.kind == "func" else self._rec_info(bt))
        if bt.kind == "func" and bt.name in self.classes:
            info = self.classes[bt.name]
            if m in info.statics:
                return self._obj_call(info.statics[m], n, n.args, n.keywords)
            if m in info.methods:
                if not n.args:
                    self.refuse(n, "`%s.%s()` without the object" % (bt.name, m), "Python raises TypeError; call obj.%s()." % m)
                return self._obj_call(info.methods[m], n, n.args, n.keywords)
            self.refuse(n, "`%s.%s`, which the class does not define" % (bt.name, m), "Python raises AttributeError.")
        info = self._rec_info(bt)
        if info is None:
            return None
        if m in info.methods:
            if self._threaded(self.funcs[info.methods[m]]) and n is not getattr(self, "_thread_site", None):
                self._threaded_refusal(n, self.funcs[info.methods[m]])
            return self._obj_call(info.methods[m], n, [n.func.value] + list(n.args), n.keywords, (base, bt, bp))
        if m in info.statics:
            if not isinstance(n.func.value, ast.Name):
                self.refuse(n, "a static method called through an expression", "Call it through the class.")
            return self._obj_call(info.statics[m], n, n.args, n.keywords)
        if info.kind == "namedtuple" and m == "_replace":
            return self._obj_replace(info, n, n.func.value, (base, bt, bp), n.args, n.keywords)
        if info.kind == "namedtuple" and m in ("_asdict", "_make", "count", "index"):
            self.refuse(n, "the named tuple method `.%s`" % m, "Read the fields by name; _replace is translated.")
        if info.kind == "dict":
            return self._dict_method(info, n, base, bt, bp)
        self.refuse(n, "the method `.%s`, which %s does not have" % (m, info.words()), "Python raises AttributeError.")

    def _threaded_refusal(self, n, f):
        thr = self._threaded(f)
        p = self.parents.get(n)
        if isinstance(p, ast.Expr) or isinstance(p, (ast.Assign, ast.Return)) and p.value is n:
            self.refuse(n, "a call of `%s`, which changes the object passed as `%s`, with an expression there" % (self._fname(f), f.params[thr[0]]),
                        "Pass a variable: the call gives the changed object back to it.")
        self.refuse(n, "a call of `%s`, which changes the object passed as `%s`, inside an expression" % (self._fname(f), f.params[thr[0]]),
                    "Call it on its own line, or as the whole right side of an assignment (`x = obj.m()`), "
                    "so the translation can give the changed object back to its variable.")

    def _obj_replace(self, info, n, recv, rv, args, keywords):
        """`p._replace(x=1)`, `dataclasses.replace(p, x=1)`: a new record, the other fields copied."""
        if args:
            self.refuse(n, "replace with a positional argument", "Name the fields: p._replace(x=1).")
        base, bt, bp = rv
        if not isinstance(recv, ast.Name):
            tmp = self.new_tmp("o")
            self.pre.append("%s = %s" % (tmp, base))
            base = tmp
        repl = {}
        order = [info.fields.index(kw.arg) for kw in keywords if kw.arg in info.fields]
        if order != sorted(order) and any(self._has_effect(kw.value) for kw in keywords):
            self.refuse(n, "replaced fields in another order than the fields', with a call among them",
                        "Python evaluates them in the order written, the record literal in the fields' order; "
                        "assign them to variables first.")
        for kw in keywords:
            if kw.arg is None or kw.arg not in info.fields:
                self.refuse(kw.value if kw.arg else n, "replacing `%s`, which %s does not have" % (kw.arg, info.words()),
                            "Python raises TypeError (ValueError for a named tuple).")
            c, t, _ = self.expr(kw.value)
            self._field_set(info, kw.arg, t, kw.value)
            repl[kw.arg] = c
        fields = ", ".join("%s: %s" % (self._fv(info, f), repl.get(f, "%s.%s" % (self._paren(base, bp, 100), self._fv(info, f))))
                           for f in info.fields)
        return "%s { %s }" % (info.viv, fields), T("rec", name=info.name), 100

    def _obj_builtin(self, n, name):
        """len of a dict or named tuple, dict(...), dataclasses.replace: None when not ours."""
        if name == "dict":
            # dict(a=1, b=2), and dict(d) of a translated dict (a copy); any other form is the
            # rule `containers` (rules/builtins.toml), which refuses it
            if len(n.args) == 1 and not n.keywords and self._rec_info(self._static_type(n.args[0])) is not None \
                    and self._rec_info(self._static_type(n.args[0])).kind == "dict":
                return self.expr(n.args[0])
            if not n.args and n.keywords and all(k.arg is not None for k in n.keywords):
                keys = tuple(k.arg for k in n.keywords)
                return self._dict_literal(n, keys, [k.value for k in n.keywords])
            return None
        if name == "len" and len(n.args) == 1 and not n.keywords:
            t = self._static_type(n.args[0])
            info = self._rec_info(t)
            if info is not None:
                if info.kind not in ("dict", "namedtuple"):
                    self.refuse(n, "len of %s" % info.words(), "Python raises TypeError (no __len__ is translated).")
                return str(len(info.fields)), NUM_I, 100
            return None
        if name in ("dataclasses.replace",) or name in self.dc_replace:
            if not n.args:
                self.refuse(n, "replace without the object", "Python raises TypeError.")
            rv = self.expr(n.args[0])
            info = self._rec_info(rv[1])
            if info is None or info.kind != "dataclass":
                self.refuse(n, "dataclasses.replace of %s" % self._kind_words(rv[1]), "Pass a dataclass object.")
            return self._obj_replace(info, n, n.args[0], rv, n.args[1:], n.keywords)
        return None

    # ---------------------------------------------------------------- dicts
    def _e_Dict(self, n):
        if not n.keys:
            self.refuse(n, "an empty dict (its keys are added at run time)",
                        "A translated dict is a literal with constant keys; a map with keys computed at run time is not in "
                        "Viv yet (use an array indexed by an int key).")
        if any(k is None for k in n.keys):
            self.refuse(n, "`**` in a dict literal", "Write the keys out.")
        keys = []
        for k in n.keys:
            v = self._dict_key(k)
            if v is None:
                self.refuse(k, "a dict key that is not a string or int constant",
                            "A translated dict has constant keys; a map with keys computed at run time is not in Viv yet.")
            keys.append(v)
        if len(set(map(repr, keys))) != len(keys):
            self.refuse(n, "a dict literal with a repeated key", "Python keeps the last value at the first key's place; write each key once.")
        return self._dict_literal(n, tuple(keys), n.values)

    def _dict_literal(self, n, keys, values):
        info = self.dict_shapes.get(keys)
        if info is None:
            self.refuse(n, "this dict", "Write a dict literal with constant keys.")
        codes = []
        for f, v in zip(info.fields, values):
            c, t, _ = self.expr(v)
            self._field_set(info, f, t, v)
            codes.append("%s: %s" % (info.fviv[f], c))
        info.used = True
        return "%s { %s }" % (info.viv, ", ".join(codes)), T("rec", name=info.name), 100

    def _dict_method(self, info, n, base, bt, bp):
        m = n.func.attr
        if m == "get":
            if not n.args or len(n.args) > 2 or n.keywords:
                self.refuse(n, "this .get", "Write d.get(key) or d.get(key, default).")
            k = self._dict_key(n.args[0])
            if k is None:
                self.refuse(n.args[0], "d.get with a key computed at run time",
                            "A translated dict has constant keys; a map is not in Viv yet.")
            f = keyattr(k)
            if f in info.fields:
                if len(n.args) == 2:
                    if self._has_effect(n.args[1]):
                        self.refuse(n.args[1], "a default of d.get that calls a def", "Python evaluates it even when the key is "
                                    "there; compute it on its own line first.")
                    self.expr(n.args[1])
                return "%s.%s" % (self._paren(base, bp, 100), info.fviv[f]), info.ftype(f), 100
            if len(n.args) == 1:
                self.refuse(n, "d.get of a key the dict does not have (None)", "`None` is not a Viv value; pass a default.")
            return self.expr(n.args[1])
        if m == "copy" and not n.args and not n.keywords:
            return base, bt, bp
        if m in ("keys", "values", "items"):
            self.refuse(n, "d.%s() outside a for loop header" % m, "Loop over it: `for k, v in d.items():` (unrolled per key).")
        self.refuse(n, "the dict method `.%s`" % m, "A translated dict keeps the keys of its literal: read, set and loop over them.")

    def _obj_subscript(self, n):
        """`p[0]` of a named tuple, `d[1]` of a dict with int keys: a field. None when not ours."""
        if not isinstance(n.value, (ast.Name, ast.Attribute)):
            return None
        t = self._static_type(n.value)
        info = self._rec_info(t)
        if info is None:
            return None
        sl = n.slice
        if info.kind == "namedtuple":
            k = None
            if isinstance(sl, ast.Constant) and type(sl.value) is int:
                k = sl.value
            elif isinstance(sl, ast.UnaryOp) and isinstance(sl.op, ast.USub) and isinstance(sl.operand, ast.Constant) \
                    and type(sl.operand.value) is int:
                k = -sl.operand.value
            if k is None:
                self.refuse(n, "an index of a named tuple that is not an int literal", "Read the field by name: p.x.")
            if not -len(info.fields) <= k < len(info.fields):
                self.refuse(n, "an index past the end of a named tuple", "Python raises IndexError.")
            f = info.fields[k]
            base, bt, bp = self.expr(n.value)
            return "%s.%s" % (self._paren(base, bp, 100), self._fv(info, f)), info.ftype(f), 100
        if info.kind == "dict":
            k = self._dict_key(sl)
            if k is None:
                self.refuse(n, "a dict read with a key computed at run time",
                            "A translated dict has constant keys; a map with keys computed at run time is not in Viv yet.")
            f = keyattr(k)
            if f not in info.fields:
                self.refuse(n, "the key %r, which %s does not have" % (k, info.words()), "Python raises KeyError.")
            base, bt, bp = self.expr(n.value)
            return "%s.%s" % (self._paren(base, bp, 100), info.fviv[f]), info.ftype(f), 100
        self.refuse(n, "indexing %s" % info.words(), "Python raises TypeError (no __getitem__ is translated).")

    def _obj_subscript_store(self, tgt, code, t, s):
        """`d[1] = v` of a dict with int keys: the field is rebound."""
        if not isinstance(tgt.value, ast.Name):
            return False
        info = self._rec_info(self._static_type(tgt.value))
        if info is None:
            return False
        if info.kind != "dict":
            self.refuse(tgt, "a store into %s" % info.words(), "Python raises TypeError.")
        k = self._dict_key(tgt.slice)
        if k is None:
            self.refuse(tgt, "a dict store with a key computed at run time", "A translated dict has constant keys.")
        att = ast.copy_location(ast.Attribute(value=tgt.value, attr=keyattr(k), ctx=ast.Store()), tgt)
        return self._obj_store(att, code, t, s)

    def _obj_in(self, n, op, right, lc, lt):
        """`"k" in d`: known from the dict's keys (its literal's)."""
        t = self._static_type(right)
        info = self._rec_info(t)
        if info is None:
            return None
        if info.kind != "dict":
            self.refuse(n, "`in` of %s" % info.words(), "Python raises TypeError, or compares with each field; compare the fields.")
        k = self._dict_key(n.left)
        if k is None:
            self.refuse(n, "`in` of a dict with a key computed at run time", "A translated dict has constant keys; compare with each key.")
        has = keyattr(k) in info.fields
        if isinstance(op, ast.NotIn):
            has = not has
        return "true" if has else "false"

    def _obj_for(self, s):
        """`for k in d`, `for k in d.keys()`, `for v in d.values()`, `for k, v in d.items()`:
        one copy of the body per key, in the literal's order (the keys are known here)."""
        it = s.iter
        what = "keys"
        recv = it
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and it.func.attr in ("keys", "values", "items") \
                and not it.args and not it.keywords:
            recv, what = it.func.value, it.func.attr
        info = self._rec_info(self._static_type(recv)) if isinstance(recv, (ast.Name, ast.Attribute)) else None
        if info is None:
            return False
        if info.kind != "dict":
            self.refuse(it, "iterating over %s" % info.words(), "Loop over its fields one by one.")
        if s.orelse:
            self.refuse(s, "`for ... else`", "Use a flag variable set before `break`.")
        for st in s.body:
            for m in ast.walk(st):
                if isinstance(m, (ast.Break, ast.Continue)):
                    self.refuse(m, "`%s` in a loop over a dict (the loop is unrolled)" % ("break" if isinstance(m, ast.Break) else "continue"),
                                "Use a flag and an if.")
        if len(info.fields) > 32:
            self.refuse(it, "a loop over a dict of more than 32 keys", "Loop over an array.")
        tgt = s.target
        if what == "items":
            if not (isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2 and all(isinstance(e, ast.Name) for e in tgt.elts)):
                self.refuse(tgt, "this loop target", "Write `for k, v in d.items():`.")
            kname, vname = tgt.elts[0].id, tgt.elts[1].id
        elif isinstance(tgt, ast.Name):
            kname, vname = (tgt.id, None) if what == "keys" else (None, tgt.id)
        else:
            self.refuse(tgt, "this loop target", "Loop over one name.")
        for f in info.fields:
            key = info.keys[f]
            self.pre = []
            body = [copy.deepcopy(b) for b in s.body]
            self.emit("# %s = %r" % (kname or vname, key), s)
            if kname is not None:
                # the key is a constant in this copy of the body (`d[k]` reads the field)
                if self._assigns_name(body, kname):
                    self.refuse(tgt, "a loop over a dict's keys that rebinds `%s`" % kname, "Use another name.")

                class Sub(ast.NodeTransformer):
                    def visit_Name(self, node):
                        if isinstance(node.ctx, ast.Load) and node.id == kname:
                            kn = ast.Constant(value=abs(key)) if not isinstance(key, str) else ast.Constant(value=key)
                            if not isinstance(key, str) and key < 0:
                                kn = ast.UnaryOp(op=ast.USub(), operand=kn)
                            return ast.copy_location(kn, node)
                        return node
                body = [Sub().visit(b) for b in body]
                if not isinstance(key, str):
                    self._assign_to(ast.copy_location(ast.Name(id=kname, ctx=ast.Store()), tgt),
                                    ast.copy_location(ast.Constant(value=key), tgt), s)
            if vname is not None:
                self.pre = []
                self._assign_to(ast.copy_location(ast.Name(id=vname, ctx=ast.Store()), tgt),
                                ast.copy_location(ast.Attribute(value=recv, attr=f, ctx=ast.Load()), it), s)
            for b in body:
                ast.fix_missing_locations(b)
            self._block(body)
        return True

    def _obj_attr_targets(self, tgt, value, s):
        """`self.a, self.b = e1, e2` (fields among the targets): Python computes the right side,
        then assigns the targets left to right. The values go to temporaries first, then each
        field is rebound in order. True when handled."""
        if not any(isinstance(x, ast.Attribute) for x in tgt.elts):
            return False
        bases = {x.value.id for x in tgt.elts if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name)}
        for x in tgt.elts:
            if isinstance(x, ast.Name) and x.id in bases:
                self.refuse(x, "an unpacking that rebinds `%s` and a field of it" % x.id, "Split it into two assignments.")
        tmps, later = [], []
        for x in tgt.elts:
            if isinstance(x, ast.Attribute):
                tmp = ast.copy_location(ast.Name(id=self.new_tmp("u"), ctx=ast.Store()), x)
                tmps.append(tmp)
                later.append((x, tmp.id))
            else:
                tmps.append(x)
        t2 = ast.copy_location(ast.Tuple(elts=tmps, ctx=ast.Store()), tgt)
        self._tuple_assign(t2, value, s)
        for x, tmp in later:
            self.pre = []
            self._store(x, self.vname(tmp), self.var_type(tmp) or UNK, s)
        return True

    def _obj_unpack(self, tgt, value, s):
        """`x, y = p` of a named tuple: each field in order."""
        if not isinstance(value, (ast.Name, ast.Attribute)):
            return False
        info = self._rec_info(self._static_type(value))
        if info is None:
            return False
        if info.kind != "namedtuple":
            self.refuse(value, "unpacking %s" % info.words(),
                        "Python unpacks a dict's keys (strings), and an object raises TypeError; read the fields.")
        names = tgt.elts
        if len(names) != len(info.fields):
            self.refuse(value, "unpacking %d fields into %d names" % (len(info.fields), len(names)),
                        "Python raises ValueError; give one name per field.")
        base, bt, bp = self.expr(value)
        codes = ["%s.%s" % (self._paren(base, bp, 100), self._fv(info, f)) for f in info.fields]
        self._unpack_into(names, ", ".join(codes), [info.ftype(f) for f in info.fields], s)
        return True

    # ---------------------------------------------------------------- printing
    def _obj_print(self, node, code, t, render, spec, parts, values):
        """An object in a print or an f-string, as CPython writes it: a dataclass or a named
        tuple `P(x=1.5, y=2)`, a dict `{'a': 1}`, a class through its __str__/__repr__ f-string."""
        info = self._rec_info(t)
        if spec is not None and spec != "":
            self.refuse(node, "a format spec on %s" % info.words(), "Python raises TypeError (object.__format__).")
        if not isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
            if self._has_effect(node):
                self.refuse(node, "printing %s made by a call in the print" % info.words(), "Assign it to a name first.")
            tmp = self.new_tmp("o")
            self.pre.append("%s = %s" % (tmp, code))
            code = tmp
        which = ["__repr__"] if render == "repr" else ["__str__", "__repr__"]
        for w in which:
            if w in info.dunders:
                fm = info.fmt.get(w)
                if fm is None:
                    self.refuse(node, "printing %s through its %s" % (info.words(), w),
                                "py2viv prints an object through a __str__ or __repr__ that returns one f-string.")
                sn, js = fm
                if isinstance(js, ast.Constant):
                    parts.append({"kind": "str", "text": js.value})
                    return
                target = node

                class Sub(ast.NodeTransformer):
                    def visit_Name(self, x):
                        if x.id == sn:
                            return copy.deepcopy(target)
                        return x
                js2 = Sub().visit(copy.deepcopy(js))
                ast.fix_missing_locations(js2)
                start = len(parts)
                self._fstring(js2, parts, values)
                for part in parts[start:]:
                    if part["kind"] == "val":
                        part["sub"] = True
                return
        if not info.auto_repr:
            self.refuse(node, "printing %s" % info.words(), "Python prints the object's address; print its attributes.")
        if info.kind == "dict":
            parts.append({"kind": "str", "text": "{"})
            for i, f in enumerate(info.fields):
                if i:
                    parts.append({"kind": "str", "text": ", "})
                parts.append({"kind": "str", "text": "%r: " % (info.keys[f],)})
                self._obj_print_field(node, code, info, f, parts, values)
            parts.append({"kind": "str", "text": "}"})
            return
        parts.append({"kind": "str", "text": info.typename + "("})
        for i, f in enumerate(getattr(info, "repr_fields", None) or info.fields):
            if i:
                parts.append({"kind": "str", "text": ", "})
            parts.append({"kind": "str", "text": f + "="})
            self._obj_print_field(node, code, info, f, parts, values)
        parts.append({"kind": "str", "text": ")"})

    def _obj_print_field(self, node, code, info, f, parts, values):
        ft = info.ftype(f)
        fnode = ast.copy_location(ast.Attribute(value=node, attr=f, ctx=ast.Load()), node)
        fcode = "%s.%s" % (code, self._fv(info, f))
        if ft.kind == "rec":
            self._obj_print(fnode, fcode, ft, "repr", None, parts, values)
            return
        # a NumPy array is written by its repr (array([...])): diff.py reads the values back from
        # the line and compares NumPy's repr of them with Python's text
        self._print_value(fnode, fcode, ft, "repr", None, parts, values)
        parts[-1]["sub"] = True  # compared as the line's text (diff.py)

    # ================================================================ calls of defs
    #
    # Two holes in the translation of calls, found while adding objects (RULES.md, "Fixed
    # holes", 2026-09-28): a default Python evaluates once, when the def runs, was evaluated at
    # every call in the caller's scope; and keyword arguments were evaluated in the order of the
    # parameters, not in the order written.

    def _const_default(self, d):
        """A default whose value is the same wherever and whenever it is evaluated: a literal,
        or a module constant (np.pi)."""
        if literal_number(d) or isinstance(d, ast.Constant):
            return True
        x = d.operand if isinstance(d, ast.UnaryOp) and isinstance(d.op, (ast.USub, ast.UAdd)) else d
        if isinstance(x, ast.Attribute) and x is d and self._ie_enum_class(x.value) is not None:
            return True  # a member of an enum (ints.py): one value, wherever it is read
        if isinstance(x, ast.Attribute):
            return self._call_name(x) in ("np.pi", "math.pi", "np.e", "math.e", "math.tau", "np.inf", "math.inf", "np.nan", "math.nan")
        if isinstance(x, ast.Name):
            return x.id in self.from_np and self.from_np[x.id] in ("pi", "e", "inf", "nan") or \
                x.id in self.from_math and self.from_math[x.id] in ("pi", "e", "tau", "inf", "nan")
        return False

    def _emit_defaults(self, f, s):
        """`def f(a, b=expr)`: Python evaluates expr once, when the def statement runs, in the
        scope around it; the translation keeps it in a module variable written there."""
        for p, d in f.defaults.items():
            if self._const_default(d):
                continue
            code, t, _ = self.expr(d)
            if t.kind in ("str", "none", "func", "module", "tuple"):
                self.refuse(d, "a default that is %s" % self._kind_words(t), "Pass numbers, bools, arrays and records.")
            info = self._rec_info(t)
            if info is not None and info.mutable:
                self.refuse(d, "a default that is %s whose fields are rebound" % info.words(),
                            "Python shares the one default object between the calls; pass the object.")
            vn = "py_dflt_%s_%s" % (viv_name(f.name), p)
            self.flush_pre(s)
            self.emit("%s = %s" % (vn, code), s)
            self._dflt[(f.name, p)] = (vn, t)

    def _default_arg(self, f, p, n):
        d = f.defaults[p]
        if self._const_default(d):
            return self.expr(d, keep=True)  # a member, an Int's literal, None of an optional (ints.py)
        r = self._dflt.get((f.name, p)) or self._prev_dflt.get((f.name, p))
        if r is None:
            self.refuse(n, "a call of `%s` that takes the default of `%s` before the def has run" % (f.name, p),
                        "Python evaluates the default when the def statement runs; call it after the def.")
        return r[0], r[1], 100

    def _has_effect(self, node):
        """Whether evaluating node may call a def of the program (which may print, write into
        an array or change an object) or draw a random number."""
        for c in ast.walk(node):
            if not isinstance(c, ast.Call):
                continue
            f = c.func
            if isinstance(f, ast.Name) and f.id in self.funcs and self._pure_ctor(self.funcs[f.id]):
                continue  # a constructor that only computes its fields (its arguments are walked too)
            if isinstance(f, ast.Name) and f.id in self.funcs or self._draws(c):
                return True
            if isinstance(f, ast.Attribute) and f.attr in self.method_keys and not self._is_module_path(f):
                return True
        return False

    PURE_CALLS = {"abs", "min", "max", "len", "float", "int", "round", "sum", "range", "bool"}

    def _pure_ctor(self, f):
        """A constructor whose __init__ calls nothing but NumPy (not np.random), math and a few
        builtins: calling it has no effect beyond its arguments'."""
        if getattr(f, "ctor_of", None) is None:
            return False
        for c in ast.walk(f.node):
            if isinstance(c, ast.Call):
                name = self._call_name(c.func) or ""
                if name.startswith("np.random") or not (name.startswith(("np.", "math.")) or name in self.PURE_CALLS):
                    return False
        return True

    def _kw_order_check(self, f, n):
        """Keyword arguments are evaluated in the order written; the translation passes them in
        the order of the parameters. Refuse where the two differ and an argument has an effect."""
        kws = [k for k in n.keywords if k.arg in f.params]
        order = [f.params.index(k.arg) for k in kws]
        if order != sorted(order) and any(self._has_effect(k.value) for k in kws):
            self.refuse(n, "keyword arguments in another order than `%s`'s parameters, with a call among them" % self._fname(f),
                        "Python evaluates them in the order written and the translation in the parameters' order; "
                        "assign them to variables first, or write them in the parameters' order.")

    def _obj_user_call(self, c):
        """A call of a method of the program (for the evaluation-order checks)."""
        f = c.func
        return isinstance(f, ast.Attribute) and f.attr in self.method_keys and not self._is_module_path(f)

    def _guess_rec_param(self, f, name):
        """A parameter no call typed (a def never called): an object, if the body reads an
        attribute (or a constant key) of it that exactly one class (or dict shape) has."""
        attrs = set()
        for n in ast.walk(f.node):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == name:
                attrs.add(n.attr)
        if not attrs:
            return None
        cands = [info for info in self._all_infos()
                 if attrs <= set(info.fields) | set(info.methods) | set(info.props) | set(info.consts)]
        if len(cands) == 1:
            return T("rec", name=cands[0].name)
        return None
