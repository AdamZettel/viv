# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""ints.py: Python ints annotated `int`, enum.Enum classes and Optional values, as Viv's Int,
enums and optionals, exactly or refused (py2viv, strict).

What is translated:

  - A parameter or a result of a module-level def annotated `int` (`def f(n: int) -> int`), and a
    variable annotated `int` (`x: int = ...`), is a Viv Int (`n: Int`, `-> Int`, `x: Int = ...`):
    signed 64-bit, every operation checked, so its arithmetic is Python's to 2^63 and an
    overflow stops the run (Python's ints do not overflow: the stop is a refusal at run time).
    `+ - * // %` of two Ints (or of an Int and a value py2viv knows is a Python int) are Viv's Int
    operators: `//` floors and `%` takes the divisor's sign, Python's rules, and a zero divisor
    stops with Python's ZeroDivisionError line. `/` of Ints is Python's true division (Viv
    converts each Int exactly, or stops past 2^53). `**` by a literal exponent of 0 or more is an
    Int power. A value py2viv keeps as a double that holds a Python int (a literal, `len(...)`,
    int arithmetic, every one integral below 2^53 by the existing guards) is converted with
    `int(...)` where it flows into an Int; an int literal is written as itself (it adapts).
    Everywhere else (a float operand, a NumPy call, an index, a range bound, a list, a
    conditional expression, ...) the Int is converted to a number exactly, `(n + 0.0)`, or the run
    stops: the translation is then the one py2viv makes for an unannotated int, which stops past
    2^53 too. A float, a string or a list where the annotation says `int` is refused; a Bool
    passed for an int parameter (Python's True is an int that prints True) gets the def's clone
    (infer.py) for a Bool, which keeps the translation py2viv made before Ints.
    Ints print, convert with str() and format in f-strings as Python's ints do (`fmt(n, "d")`).
    Unannotated ints keep their translation (doubles), so programs without annotations are
    translated as before.
  - A class deriving from enum.Enum (`class C(enum.Enum)` or `from enum import Enum`) whose body
    is assignments of distinct int or str constants to names (and a docstring) is a payload-free
    Viv enum: `enum C { A, B }`. `C.A` is `C.A`; `==`, `!=`, `is` and `is not` between members of
    one enum are `==` and `!=`; `print(m)`, `str(m)` and an f-string field write Python's text
    `C.A` (Viv's str of an enum is the variant's name, so the translation writes "C." before
    it); `m.name` is the variant's name. A member may be bound to a name, passed to a def,
    returned, compared, printed and converted to text; nothing else.
  - `Optional[T]`, `T | None` and `Union[T, None]` on a parameter, a result or a variable, for T
    one of int (a Viv `Int?`), float (`Num?`), str (`Str?`), bool (`Bool?`) or an enum class, is
    Viv's `T?`. `None` is `None`; a T value stored, passed or returned where an optional is
    declared is `Some(v)`; a def whose result is optional and that reaches its end returns
    `None`, as Python does. An optional is taken apart only by
    `if x is None: A else: B` and `if x is not None: B else: A` (x a variable): the translation
    is `match x { None => { A } Some(v) => { B, with x read as v } }`.

What is refused, with the reason: `.value` of a member, `auto()`, a member looked up by value
or name (`C(1)`, `C["A"]`), iterating an enum or using the class as a value, methods and
anything but constant members in an enum's body, equal values (aliases), IntEnum, Flag,
StrEnum and IntFlag, a member in arithmetic, in a list, a dict, a tuple or a record (Viv's
arrays, maps and records do not hold enums yet), the truth value of a member; an optional used
in any other way (arithmetic, printing, `x is None` as a value or beside other conditions, a
narrowing that continues after its `if`, assigning x where it is known not to be None); a
float where Optional[int] is declared (Python keeps the int and prints it as an int, `3` and not
`3.0`, so a Some of a number would print differently: the declared and the passed types must
agree); an int literal past 2^53 (as before); an int-annotated def used as a value; a def
annotated `-> int` that can reach its end (Python returns None there).
"""

import ast
import re

P = None  # the py2viv module (set by py2viv.py)

IE_KINDS = ("int", "enum", "opt")

# names an enum or one of its members may not take: Viv's keywords, and Viv's type names
VIV_TYPE_NAMES = {"Int", "Num", "Str", "Bool", "Arr", "CArr", "Map", "Some", "None"}

ENUM_BASES_REFUSED = {
    "IntEnum": "an IntEnum (its members are ints: arithmetic, comparison with numbers, int values)",
    "IntFlag": "an IntFlag (bit sets of int members)",
    "Flag": "a Flag (bit sets of members)",
    "StrEnum": "a StrEnum (its members are strings)",
    "ReprEnum": "a ReprEnum",
    "EnumMeta": "an enum metaclass",
    "EnumType": "an enum metaclass",
}


def enum_t(name):
    return P.T("enum", name=name)


def opt_t(inner):
    return P.T("opt", elts=[inner])


def is_int_num(t):
    """A double py2viv keeps that holds a Python int (integral, below 2^53 by its guards)."""
    return t is not None and t.kind == "num" and t.py == "int"


def join_var(a, b):
    """The type of a variable bound to a and to b: py2viv's join, and an Int beside a Python int
    kept as a double is an Int (the double is converted where it is stored), and an optional
    beside None or its own value type is that optional."""
    if a is not None and b is not None:
        if a.kind == "int" and (b.kind == "int" or is_int_num(b)) or b.kind == "int" and is_int_num(a):
            return P.INT
        for x, y in ((a, b), (b, a)):
            if x.kind == "opt" and (y.kind == "none" or y == x.elts[0] or x.elts[0].kind == "int" and is_int_num(y)):
                return x
    return P.join(a, b)


def int_literal(code):
    return re.fullmatch(r"-?\d+", code) is not None


class EnumInfo:
    def __init__(self, name, node):
        self.name = name
        self.node = node
        self.members = []
        self.refusal = None
        self.viv = P.viv_name(name)

    def refuse(self, node, what, hint):
        if self.refusal is None:
            self.refusal = (node, what, hint)


class IntsMixin:
    """The translator's handling of int annotations, enums and optionals (see the docstring)."""

    # ================================================================ the module scan
    def _ie_scan_enums(self):
        """The enum classes at module level (and under the main guard)."""
        self.enums = {}
        self.enum_nodes = {}
        self.enum_mods = set()     # names bound to the module enum
        self.enum_names = {}       # local name -> what it is in the module enum (Enum, IntEnum, auto, ...)
        tops = list(self.tree.body)
        for node in self.tree.body:
            if isinstance(node, ast.If) and self._is_main_guard(node):
                tops += node.body
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name == "enum":
                        self.enum_mods.add(a.asname or a.name)
            elif isinstance(node, ast.ImportFrom) and node.module == "enum":
                for a in node.names:
                    self.enum_names[a.asname or a.name] = a.name
        for node in tops:
            if isinstance(node, ast.ClassDef):
                base = self._ie_enum_base(node)
                if base is not None:
                    info = self._ie_analyze_enum(node, base)
                    self.enums[node.name] = info
                    self.enum_nodes[id(node)] = info

    def _ie_enum_base(self, c):
        """What the class derives from in the module enum (Enum, IntEnum, ...), or None."""
        for b in c.bases:
            if isinstance(b, ast.Name) and b.id in self.enum_names:
                return self.enum_names[b.id]
            if isinstance(b, ast.Attribute) and isinstance(b.value, ast.Name) and b.value.id in self.enum_mods:
                return b.attr
        return None

    def _ie_analyze_enum(self, c, base):
        info = EnumInfo(c.name, c)
        if base != "Enum":
            info.refuse(c, ENUM_BASES_REFUSED.get(base, "a class deriving from enum.%s" % base),
                        "py2viv translates a plain enum.Enum whose members are constants.")
        if len(c.bases) != 1:
            info.refuse(c, "an enum with several base classes", "Derive from enum.Enum alone.")
        if c.keywords:
            info.refuse(c, "an enum class with `%s=`" % c.keywords[0].arg, "Derive from enum.Enum alone.")
        if c.decorator_list:
            info.refuse(c.decorator_list[0], "a decorator on an enum class (`@%s`)" % ast.unparse(c.decorator_list[0]),
                        "Drop it: py2viv translates a plain enum.Enum, whose members are distinct already.")
        if c.name in VIV_TYPE_NAMES or c.name in P.VIV_KEYWORDS:
            info.refuse(c, "an enum named `%s`, a Viv type name or keyword" % c.name, "Rename the class.")
        body = c.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body = body[1:]
        values = {}
        for st in body:
            if isinstance(st, ast.Pass):
                continue
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                info.refuse(st, "a method in an enum (`%s`)" % st.name,
                            "Viv's enums have no methods: write a def that takes the member.")
                continue
            if not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)):
                info.refuse(st, "a statement in an enum's body other than `NAME = constant`",
                            "py2viv translates an enum whose members are assigned int or str constants.")
                continue
            name = st.targets[0].id
            v = st.value
            if isinstance(v, ast.Call) and (isinstance(v.func, ast.Name) and self.enum_names.get(v.func.id) == "auto"
                                            or isinstance(v.func, ast.Attribute) and v.func.attr == "auto"):
                info.refuse(v, "`auto()`", "Write the values: A = 1, B = 2, ...; py2viv checks they are distinct.")
                continue
            if isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.USub) and isinstance(v.operand, ast.Constant) \
                    and type(v.operand.value) is int:
                val = -v.operand.value
            elif isinstance(v, ast.Constant) and type(v.value) in (int, str):
                val = v.value
            else:
                info.refuse(v, "an enum member whose value is not an int or str constant",
                            "py2viv translates members assigned int or str constants (the values are not used).")
                continue
            if name.startswith("_"):
                info.refuse(st, "an enum member named `%s` (a leading underscore)" % name,
                            "Python treats _sunder_ and __dunder__ names as settings and _private names as non-members; "
                            "rename it.")
                continue
            if name in P.VIV_KEYWORDS or name in VIV_TYPE_NAMES:
                info.refuse(st, "an enum member named `%s`, a Viv keyword or type name" % name, "Rename it.")
                continue
            if (type(val), val) in values:
                info.refuse(st, "the enum member `%s`, whose value equals `%s`'s (an alias)" % (name, values[(type(val), val)]),
                            "Python makes it another name for the same member; give each member its own value.")
                continue
            if name in info.members:
                info.refuse(st, "the enum member `%s` assigned twice" % name, "Python raises TypeError.")
                continue
            values[(type(val), val)] = name
            info.members.append(name)
        if not info.members:
            info.refuse(c, "an enum without members", "Viv's enum has at least one variant.")
        return info

    def _ie_scan_decls(self):
        """The declared types: each scope's variables annotated int, Optional[...] or an enum,
        and the parameters and results of the defs whose annotations are followed."""
        self.ie_decls = {}
        self.ie_decl_conflicts = []

        def scan(key, body, params=()):
            d = {}
            for p, ann, node in params:
                t = self._ie_ann(ann)
                if t is not None:
                    d[p] = t
            stack = list(body)
            while stack:
                st = stack.pop(0)
                if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    continue
                if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                    t = self._ie_ann(st.annotation)
                    if t is not None:
                        old = d.get(st.target.id)
                        if old is not None and old != t:
                            self.ie_decl_conflicts.append((st, st.target.id))
                        d[st.target.id] = t
                for c in ast.iter_child_nodes(st):
                    if isinstance(c, ast.stmt):
                        stack.append(c)
                    elif isinstance(c, ast.excepthandler):
                        stack.extend(c.body)
            self.ie_decls[key] = d

        scan("<module>", [st for st in self.tree.body])
        for f in self.funcs.values():
            if not self._ie_follows(f):
                continue
            args = f.node.args.args
            scan(id(f.node), f.node.body, [(a.arg, a.annotation, a) for a in args])

    def _ie_follows(self, f):
        """Whether py2viv follows this def's int and Optional annotations: a plain module-level
        def (not a method, not differentiated by grad), and not the clone (infer.py) of a def
        whose parameter annotated int is passed a Bool (Python's True is an int that prints
        True: that clone keeps the translation py2viv made before Ints)."""
        b = getattr(f, "base", f)
        return bool(getattr(b, "plain", False)) and b.name not in getattr(self, "grad_defs", set()) \
            and not b.node.decorator_list and not self._ie_bool_variant(f)

    def _ie_bool_variant(self, f):
        types = getattr(f, "param_types", None)
        if not types:
            return False
        args = getattr(f, "base", f).node.args.args
        for i, a in enumerate(args):
            if i < len(types) and self._ie_ann(a.annotation) == P.INT:
                t = P.join(types[i], f.prev_params[i]) if i < len(f.prev_params) else types[i]
                if t is not None and t.kind == "bool":
                    return True
        return False

    def _ie_ann(self, ann):
        """The type an annotation declares, as this module follows it: int (a Viv Int), an enum
        class, an Optional of int, float, str, bool or an enum. None for anything else."""
        if ann is None:
            return None
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            try:
                ann = ast.parse(ann.value, mode="eval").body
            except SyntaxError:
                return None
        base = self._ie_ann_base(ann)
        if base is not None and base.kind in ("int", "enum"):
            return base
        inner = self._ie_optional_inner(ann)
        if inner is not None:
            t = self._ie_ann_base(inner)
            if t is not None:
                return opt_t(t)
        return None

    def _ie_ann_base(self, ann):
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            try:
                ann = ast.parse(ann.value, mode="eval").body
            except SyntaxError:
                return None
        if isinstance(ann, ast.Name):
            if ann.id in getattr(self, "enums", {}) and self.enums[ann.id].refusal is None:
                return enum_t(ann.id)
            return {"int": P.INT, "float": P.NUM_F, "str": P.STR, "bool": P.BOOL}.get(ann.id)
        return None

    @staticmethod
    def _ie_is_none(x):
        return isinstance(x, ast.Constant) and x.value is None

    def _ie_optional_inner(self, ann):
        """T of Optional[T], Union[T, None], Union[None, T], T | None, None | T."""
        if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
            if self._ie_is_none(ann.right):
                return ann.left
            if self._ie_is_none(ann.left):
                return ann.right
            return None
        if isinstance(ann, ast.Subscript):
            v = ann.value
            name = v.id if isinstance(v, ast.Name) else (v.attr if isinstance(v, ast.Attribute) else None)
            sl = ann.slice
            if name == "Optional":
                return sl
            if name == "Union" and isinstance(sl, ast.Tuple) and len(sl.elts) == 2:
                a, b = sl.elts
                if self._ie_is_none(b):
                    return a
                if self._ie_is_none(a):
                    return b
        return None

    def _ie_decl(self, name):
        """The declared type of a variable or parameter of the current scope, or None."""
        decls = getattr(self, "ie_decls", None)
        if not decls or self._comp_get(name) is not None:
            return None
        sk = self._scope_of(name)
        if sk == "<module>":
            return decls.get("<module>", {}).get(name)
        if self.scope is None or not self._ie_follows(self.scope):
            return None
        return decls.get(id(self.scope.node), {}).get(name)

    def _ie_param_decl(self, f, i):
        if not self._ie_follows(f):
            return None
        decls = getattr(self, "ie_decls", {}).get(id(f.node))
        if decls is None:
            return None
        return decls.get(f.params[i])

    def _ie_ret_decl(self, f):
        if not self._ie_follows(f):
            return None
        base = getattr(f, "base", f)
        return self._ie_ann(base.node.returns)

    def _ie_typed_def(self, f):
        """A def with a parameter or result this module declares (called by name only)."""
        if not self._ie_follows(f):
            return False
        return self._ie_ret_decl(f) is not None or any(self._ie_param_decl(f, i) is not None for i in range(len(f.params)))

    # ================================================================ types as Viv writes them
    def _ie_viv_type(self, t):
        if t.kind == "int":
            return "Int"
        if t.kind == "num":
            return "Num"
        if t.kind == "str":
            return "Str"
        if t.kind == "bool":
            return "Bool"
        if t.kind == "enum":
            return self.enums[t.name].viv
        if t.kind == "opt":
            return self._ie_viv_type(t.elts[0]) + "?"
        return None

    def _ie_annot(self, t):
        if t is not None and t.kind in IE_KINDS:
            return ": " + self._ie_viv_type(t)
        return None

    def _ie_words(self, t):
        if t.kind == "int":
            return "an Int (an int-annotated value)"
        if t.kind == "enum":
            return "a member of the enum `%s`" % t.name
        if t.kind == "opt":
            inner = t.elts[0]
            return "an Optional[%s]" % (inner.name if inner.kind == "enum" else {"int": "int", "num": "float", "str": "str", "bool": "bool"}[inner.kind])
        return None

    def _ie_ann_text(self, t):
        if t.kind == "int":
            return "int"
        if t.kind == "enum":
            return t.name
        if t.kind == "opt":
            return "Optional[%s]" % self._ie_ann_text(t.elts[0])
        return {"num": "float", "str": "str", "bool": "bool"}.get(t.kind, t.kind)

    # ================================================================ values outside their places
    def _ie_lower(self, node, r):
        """An expression's (code, type, prec) where an Int, an enum member or an optional is not
        expected: an Int becomes a number exactly (`n + 0.0` converts it exactly or stops the
        run), so the rest of py2viv translates it as a Python int kept as a double; a member or
        an optional is refused."""
        code, t, prec = r[0], r[1], r[2]
        if t is None:
            return r
        if t.kind == "int":
            return "(%s + 0.0)" % code, P.NUM_I, 100
        if t.kind == "enum":
            self.refuse(node, "a member of the enum `%s` used here" % t.name,
                        "py2viv translates an enum member bound to a name, passed to a def, returned, compared with "
                        "== != is or is not, printed, and its .name; Viv's arrays, maps, tuples and records do not hold "
                        "enums yet, and a member is not a number.")
        if t.kind == "opt":
            self.refuse(node, "%s value used here (it may be None)" % self._ie_words(t),
                        "Read it inside `if x is not None:` or the else of `if x is None:` (py2viv translates that "
                        "test into a match); code after an `if x is None: return ...` does not narrow it.")
        if t.kind == "func" and t.name in self.funcs and self._ie_typed_def(self.funcs[t.name]):
            self.refuse(node, "the def `%s`, whose annotations py2viv follows (int or Optional), used as a value" % t.name,
                        "Call it by name.")
        return r

    def _ie_low_int(self, code, t, prec):
        """An Int as a number (exactly, or the run stops), for an operation py2viv translates on
        Python ints kept as doubles."""
        if t.kind == "int":
            return "(%s + 0.0)" % code, P.NUM_I, 100
        return code, t, prec

    def _ie_int_code(self, code, t, prec=100):
        """An Int, or a Python int kept as a double converted to one: (code, prec)."""
        if t.kind == "int":
            return code, prec
        if int_literal(code):
            return code, (100 if not code.startswith("-") else 80)
        return "int(%s)" % code, 100

    # ================================================================ binding: stores, arguments, results
    def _ie_conv(self, decl, code, t, node, what):
        """code of type t bound where decl is declared: (code, decl), converted, or refused."""
        if t.kind == "unknown":
            return code, decl
        if decl.kind == "int":
            if t.kind == "int":
                return code, decl
            if is_int_num(t):
                return self._ie_int_code(code, t)[0], decl
            self.refuse(node, self._ie_mismatch(t, what),
                        "The annotation and the call disagree; py2viv follows neither silently. Fix one of them." if callable(what) else
                        "Python keeps the value as it is; the annotation and the value disagree, and py2viv follows "
                        "neither silently. Convert it with int(...), or fix the annotation.")
        if decl.kind == "enum":
            if t == decl:
                return code, decl
            self.refuse(node, self._ie_mismatch(t, what), "Pass a member of `%s`." % decl.name)
        if decl.kind == "opt":
            if t.kind == "none":
                return "None", decl
            if t == decl:
                return code, decl
            if t.kind == "opt":
                self.refuse(node, self._ie_mismatch(t, what), "The two optionals hold different types.")
            inner = decl.elts[0]
            if inner.kind in ("int", "enum"):
                c, _ = self._ie_conv(inner, code, t, node, what)
                return "Some(%s)" % c, decl
            ok = t.kind == inner.kind and (t.kind != "num" or t.py == "float")
            if not ok:
                self.refuse(node, self._ie_mismatch(t, what),
                            "Python keeps the value as it is (an int stays an int and prints as one); pass a value of the "
                            "declared type." if is_int_num(t) or t.kind == "int" else
                            "Pass a value of the declared type, or None.")
            return "Some(%s)" % code, decl
        return code, t

    def _ie_mismatch(self, t, what):
        """The refusal's text for a value of type t bound where `what` declares another type: what
        is a phrase ("`x` is annotated `int`") or a function of the value's words."""
        return what(self._ie_value_words(t)) if callable(what) else "%s where %s" % (self._ie_value_words(t), what)

    def _ie_value_words(self, t):
        if t.kind == "num":
            return {"int": "an int", "float": "a float", "mixed": "a number that is an int on some paths and a float on others"}.get(t.py, "a number")
        if t.kind == "none":
            return "None"
        if t.is_arr():
            return "a %s" % ("Python list" if t.container == "list" else "NumPy array")
        return self._ie_words(t) or self._kind_words(t)

    def _ie_store(self, tgt, code, t, s):
        """`x = value` of a variable declared int, Optional or an enum, or bound to an Int, a
        member or an optional: True when stored here."""
        name = tgt.id
        env = getattr(self, "_ie_narrowed", {}).get(name)
        if env is not None and name in env:
            # x = e where x is known not to be None: a store into the optional itself (_ie_narrow)
            held = env.pop(name)
            try:
                return self._ie_store(tgt, code, t, s)
            finally:
                env[name] = held
        decl = self._ie_decl(name)
        cur = self.var_type(name)
        if decl is None and t.kind not in IE_KINDS and not (cur is not None and cur.kind in ("int", "opt")):
            return False
        if decl is not None:
            code, t = self._ie_conv(decl, code, t, getattr(s, "value", None) or s,
                                    "`%s` is annotated `%s`" % (name, self._ie_ann_text(decl)))
        elif cur is not None and cur.kind == "opt" and (t.kind == "none" or join_var(cur, t) == cur):
            code, t = self._ie_conv(cur, code, t, getattr(s, "value", None) or s, "`%s` holds %s" % (name, self._ie_words(cur)))
        self._ie_storing = True
        try:
            self.set_var(name, t, s)
        finally:
            self._ie_storing = False
        vt = self.var_type(name)
        if vt is not None and vt.kind == "int" and is_int_num(t):
            code = self._ie_int_code(code, t)[0]
        annot = self._ie_annot(vt) if vt is not None and vt.kind in ("int", "opt") else None
        self.flush_pre(s)
        self.emit("%s%s = %s" % (self.vname(name), annot or "", code), s)
        return True

    def _ie_parallel(self, names, vals, safe, s):
        """`a, b = x, y` where a value is an Int, a member or an optional, or a target holds one:
        each value into a variable first when a later value reads an earlier target (a swap),
        then each stored as `=` stores it. False when none is involved."""
        if not any(r[1].kind in IE_KINDS for r in vals) and not any(
                (self._ie_decl(n.id) or self.var_type(n.id) or P.UNK).kind in IE_KINDS for n in names):
            return False
        codes = []
        for (code, t, _), n in zip(vals, names):
            if not safe:
                if t.kind == "none":
                    self.refuse(n, "None in a parallel assignment that swaps values", "Assign it on its own line.")
                tmp = self.new_tmp("u")
                self.flush_pre(s)
                annot = self._ie_annot(t) if t.kind in ("int", "opt") else None
                self.emit("%s%s = %s" % (tmp, annot or "", code), s)
                code = tmp
            codes.append((code, t))
        for (code, t), n in zip(codes, names):
            self._store(n, code, t, s)
        return True

    def _ie_set_check(self, name, t, node):
        """set_var of a variable that holds an Int or an optional, from a binding this module did
        not convert (a loop variable, an unpacked value): refused."""
        if getattr(self, "_ie_storing", False) or t is None or t.kind in ("unknown",):
            return
        vt = self._ie_decl(name) or self.var_type(name)
        if vt is not None and vt.kind in ("int", "opt") and t != vt:
            self.refuse(node, "`%s`, which holds %s, bound here to %s" % (name, self._ie_words(vt), self._ie_value_words(t)),
                        "py2viv converts a value stored by `=` into an Int or an optional; a loop variable or an unpacked "
                        "value is not converted: use another name.")

    def _ie_arg(self, f, i, p, c, t, node):
        """An argument for parameter i of def f, converted to its declared type."""
        decl = self._ie_param_decl(f, i)
        if decl is not None and decl.kind == "int" and t.kind == "bool":
            return c, t  # a Bool for an int parameter: the def's clone for it keeps the old translation
        if decl is not None:
            ann = ast.unparse(getattr(f, "base", f).node.args.args[i].annotation)
            # worded as infer.py's _check_hints words a call that contradicts an annotation
            return self._ie_conv(decl, c, t, node, lambda tw: "`%s` called with %s for `%s`, which is annotated `%s`" % (
                f.name, tw, p, ann))
        if t.kind in IE_KINDS and (not getattr(getattr(f, "base", f), "plain", False) or f.name in getattr(self, "classes", {})):
            # a method, a constructor: its parameters are record fields or untyped (classes.py)
            r = self._ie_lower(node, (c, t, 100))
            return r[0], r[1]
        if t.kind == "func" and t.name in self.funcs and self._ie_typed_def(self.funcs[t.name]):
            self._ie_lower(node, (c, t, 100))
        return c, t

    def _ie_return(self, s):
        """`return ...` in a def whose result is declared, or that returns an Int, a member or an
        optional: True when emitted here."""
        f = self.scope
        decl = self._ie_ret_decl(f)
        v = s.value
        none = v is None or isinstance(v, ast.Constant) and v.value is None
        if decl is None:
            return False
        if none:
            if decl.kind != "opt":
                self.refuse(s, "returning None from `%s`, annotated `-> %s`" % (f.name, self._ie_ann_text(decl)),
                            "Return a value of the declared type, or annotate the result Optional[...].")
            self.flush_pre(s)
            f.ret = join_var(f.ret, decl) or decl
            self.emit("return None", s)
            return True
        if isinstance(v, (ast.Tuple, ast.IfExp)):
            if isinstance(v, ast.Tuple):
                self.refuse(v, "returning several values from `%s`, annotated `-> %s`" % (f.name, self._ie_ann_text(decl)),
                            "Return one value of the declared type.")
            return False  # split into two returns by the caller (py2viv.py)
        code, t0, _ = self.expr(v, keep=True)
        code, t = self._ie_conv(decl, code, t0, v, "`%s`'s result is annotated `-> %s`" % (f.name, self._ie_ann_text(decl)))
        f.ret = decl
        self._note_return(v, t)
        self.flush_pre(s)
        self.emit("return " + code, s)
        return True

    def _ie_ret_int(self, f, code, t, s):
        """`return v` of a def without a declared result that returns an Int elsewhere: a Python
        int kept as a double is converted (exact), through a typed variable in a recursive def:
        the def's result type is inferred, and the recursion limit's temporary takes a declared
        type only (lang/viv/src/recursion.rs)."""
        if not is_int_num(t):
            return code, t
        code = self._ie_int_code(code, t)[0]
        if self._ie_recursive(f):
            tmp = self.new_tmp("r")
            self.flush_pre(s)
            self.emit("%s: Int = %s" % (tmp, code), s)
            code = tmp
        return code, P.INT

    def _ie_recursive(self, f):
        """Whether the def calls itself, directly or through other defs of the program."""
        base = getattr(f, "base", f)
        memo = getattr(self, "_ie_rec_memo", None)
        if memo is None:
            memo = self._ie_rec_memo = {}
        if base.name in memo:
            return memo[base.name]
        calls = {}
        for k, g in self.funcs.items():
            calls[k] = {m.func.id for m in ast.walk(g.node) if isinstance(m, ast.Call) and isinstance(m.func, ast.Name)
                        and m.func.id in self.funcs}
        seen, stack = set(), list(calls.get(base.name, ()))
        while stack:
            g = stack.pop()
            if g in seen:
                continue
            seen.add(g)
            stack.extend(calls.get(g, ()))
        memo[base.name] = base.name in seen
        return memo[base.name]

    @staticmethod
    def _ie_terminal(body):
        """Whether control cannot run past the end of these statements."""
        if not body:
            return False
        last = body[-1]
        if isinstance(last, (ast.Return, ast.Raise)):
            return True
        if isinstance(last, ast.If):
            return bool(last.orelse) and IntsMixin._ie_terminal(last.body) and IntsMixin._ie_terminal(last.orelse)
        if isinstance(last, ast.While) and isinstance(last.test, ast.Constant) and last.test.value is True:
            def breaks(stmts):
                for st in stmts:
                    if isinstance(st, ast.Break):
                        return True
                    if isinstance(st, (ast.For, ast.While, ast.FunctionDef, ast.ClassDef)):
                        continue
                    if breaks([c for c in ast.iter_child_nodes(st) if isinstance(c, ast.stmt)]):
                        return True
                return False
            return not breaks(last.body)
        return False

    def _ie_def_end(self, f, s, body):
        """At the end of a def's body: a def whose result is optional returns None there, as
        Python does; one annotated `-> int` that can get there is refused."""
        decl = self._ie_ret_decl(f)
        if decl is None or self._ie_terminal(body):
            return
        if decl.kind == "opt":
            self.emit("return None", s)
            return
        self.refusals.append(P.Refusal(s, "`%s`, annotated `-> %s`, can reach its end without `return`" % (f.name, self._ie_ann_text(decl)),
                                       "Python returns None there; end every path with a return (or raise), or annotate "
                                       "the result Optional[...]."))

    # ================================================================ operators
    def _ie_binop(self, n, op, a, ta, pa, b, tb, pb):
        """`a op b` with an Int operand: (code, type, prec), or None to convert the Ints to numbers
        and translate it as py2viv does Python ints kept as doubles."""
        if ta.kind != "int" and tb.kind != "int":
            return None
        both = all(t.kind == "int" or is_int_num(t) for t in (ta, tb))
        if not both:
            return None
        if isinstance(op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Div)):
            a2, qa = self._ie_int_code(a, ta, pa)
            b2, qb = self._ie_int_code(b, tb, pb)
            if isinstance(op, (ast.FloorDiv, ast.Mod, ast.Div)) and not self._nonzero_literal(n.right):
                fn = {ast.FloorDiv: "py_ifloordiv", ast.Mod: "py_imod", ast.Div: "py_idiv"}[type(op)]
                return "%s(%s, %s)" % (self.use(fn), a2, b2), (P.NUM_F if isinstance(op, ast.Div) else P.INT), 100
            prec = 70 if isinstance(op, (ast.Mult, ast.FloorDiv, ast.Mod, ast.Div)) else 60
            return "%s %s %s" % (self._paren(a2, qa, prec), self._opsym(op), self._paren(b2, qb, prec + 1)), \
                (P.NUM_F if isinstance(op, ast.Div) else P.INT), prec
        if isinstance(op, ast.Pow) and ta.kind == "int" and isinstance(n.right, ast.Constant) \
                and type(n.right.value) is int and n.right.value >= 0:
            return "%s ^ %s" % (self._paren(a, pa, 91), n.right.value), P.INT, 90
        return None

    def _ie_unary(self, n, a, t, p):
        if t.kind == "int" and isinstance(n.op, ast.USub):
            return "-%s" % self._paren(a, p, 80), P.INT, 80
        if t.kind == "int" and isinstance(n.op, ast.UAdd):
            return a, t, p
        if t.kind in ("enum", "opt") and not isinstance(n.op, ast.Not):
            self._ie_lower(n.operand, (a, t, p))
        return None

    def _ie_truth(self, node, code, t, prec):
        if t.kind == "int":
            return "%s != 0" % self._paren(code, prec, 51)
        if t.kind == "enum":
            self.refuse(node, "the truth value of a member of the enum `%s`" % t.name,
                        "A member is always true in Python; compare it with == instead.")
        self.refuse(node, "the truth value of %s" % self._ie_words(t),
                    "Write `if x is not None:` or `if x is None:` (py2viv translates that test into a match).")

    def _ie_compare(self, n, op, lc, lt, lp, rc, rt, rp):
        """One comparison of a chain whose operand is an Int, a member or an optional: (lc, lt,
        lp, rc, rt, rp, part) with part the Viv comparison, or part None to go on with the Ints
        converted to numbers."""
        kinds = (lt.kind, rt.kind)
        if "opt" in kinds:
            t = lt if lt.kind == "opt" else rt
            if isinstance(op, (ast.Is, ast.IsNot)):
                self.refuse(n, "an `is None` test of %s used as a value or beside other conditions" % self._ie_words(t),
                            "py2viv translates `if x is None: ... else: ...` and `if x is not None: ...` on a variable, "
                            "alone in its if, into a match.")
            self._ie_lower(n, ("", t, 100))
        if "enum" in kinds:
            if lt.kind == "enum" and rt.kind == "enum" and lt == rt and isinstance(op, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)):
                sym = "==" if isinstance(op, (ast.Eq, ast.Is)) else "!="
                return lc, lt, lp, rc, rt, rp, "%s %s %s" % (self._paren(lc, lp, 51), sym, self._paren(rc, rp, 51))
            if lt.kind == "enum" and rt.kind == "enum" and lt != rt:
                self.refuse(n, "comparing members of two enums (`%s` and `%s`)" % (lt.name, rt.name),
                            "Python's answer is always False (== and is) or True (!=); compare members of one enum.")
            if lt.kind == "enum" and rt.kind == "enum":
                self.refuse(n, "ordering enum members (`%s`)" % ast.unparse(n), "Python raises TypeError; compare with == or !=.")
            other = rt if lt.kind == "enum" else lt
            self.refuse(n, "comparing a member of the enum `%s` with %s" % ((lt if lt.kind == "enum" else rt).name, self._kind_words(other)),
                        "Python's == between a plain Enum member and another value is False; compare members with members.")
        if isinstance(op, (ast.Is, ast.IsNot)):
            lc, lt, lp = self._ie_low_int(lc, lt, lp)
            rc, rt, rp = self._ie_low_int(rc, rt, rp)
            return lc, lt, lp, rc, rt, rp, None
        if all(t.kind in ("int", "num") for t in (lt, rt)):
            # Viv compares an Int with a number exactly (it never stops), as Python does
            return lc, lt, lp, rc, rt, rp, "%s %s %s" % (self._paren(lc, lp, 51), {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">",
                                                         ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}[type(op)],
                                                         self._paren(rc, rp, 51))
        lc, lt, lp = self._ie_low_int(lc, lt, lp)
        rc, rt, rp = self._ie_low_int(rc, rt, rp)
        return lc, lt, lp, rc, rt, rp, None

    def _ie_builtin(self, n, name):
        """str(x), repr(x) of an Int or a member; abs, min, max, round, int and float of Ints, as
        Viv's Int forms (exact: abs, min, max, round and int of Ints are Ints, and float(n) is the
        nearest double, Python's rule). None to go on to the rules, with every argument
        translated once (kept for them in the memo)."""
        if name not in ("str", "repr", "abs", "min", "max", "float", "int", "round") or n.keywords or not n.args \
                or any(isinstance(a, ast.Starred) for a in n.args) or not self._no_def(name):
            return None
        if not any(self._ie_peek(a) is not None for a in n.args):
            return None
        if name in ("str", "repr", "abs", "float", "int", "round") and len(n.args) != 1:
            return None
        rs = [self.expr(a, keep=True) for a in n.args]
        if name in ("str", "repr"):
            return self._str_of(n.args[0], rs[0][0], rs[0][1], rs[0][2], name)
        if any(r[1].kind == "int" for r in rs) and all(r[1].kind == "int" or is_int_num(r[1]) for r in rs):
            codes = [self._ie_int_code(r[0], r[1])[0] for r in rs]
            if name == "float":
                return "float(%s)" % codes[0], P.NUM_F, 100
            if name in ("int", "round"):
                return codes[0], P.INT, 100 if rs[0][2] >= 100 else rs[0][2]
            return "%s(%s)" % (name, ", ".join(codes)), P.INT, 100
        for a, r in zip(n.args, rs):
            self._xmemo[id(a)] = self._ie_lower(a, r)
        return None

    def _ie_big_literal_ok(self, n):
        """An int literal past 2^53 that stands where an Int is bound or computed, so that Viv
        reads it as an Int (exactly): the value of a variable, a parameter or a result declared
        int, or an operand of + - * // % beside an Int."""
        node, p = n, self.parents.get(n)
        if isinstance(p, ast.UnaryOp) and isinstance(p.op, ast.USub):
            node, p = p, self.parents.get(p)
        if isinstance(p, (ast.Assign, ast.AnnAssign)) and p.value is node:
            tg = p.targets[0] if isinstance(p, ast.Assign) and len(p.targets) == 1 else getattr(p, "target", None)
            if isinstance(tg, ast.Name):
                t = self._ie_decl(tg.id) or self.var_type(tg.id)
                return t is not None and t.kind == "int"
            return False
        if isinstance(p, ast.Return) and p.value is node and self.scope is not None:
            d = self._ie_ret_decl(self.scope)
            return d is not None and d.kind == "int"
        if isinstance(p, ast.keyword):
            p = self.parents.get(p)
        if isinstance(p, ast.Call) and isinstance(p.func, ast.Name) and p.func.id in self.funcs and node is not p.func:
            f = self.funcs[p.func.id]
            if node in p.args:
                i = p.args.index(node)
            else:
                names = [k.arg for k in p.keywords if k.value is node]
                i = f.params.index(names[0]) if names and names[0] in f.params else None
            d = self._ie_param_decl(f, i) if i is not None and i < len(f.params) else None
            return d is not None and d.kind == "int"
        if isinstance(p, ast.BinOp) and isinstance(p.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
            other = p.right if p.left is node else p.left
            return self._ie_peek(other) == "int"
        return False

    def _ie_peek(self, node):
        """The kind (int, enum, opt) an expression has by its form, without translating it: a
        variable, a member, a call of a def whose result is declared, arithmetic or a negation
        of those. None when it is none of these (or not known by its form)."""
        if isinstance(node, ast.Name):
            t = self.var_type(node.id)
            return t.kind if t is not None and t.kind in IE_KINDS else None
        if isinstance(node, ast.Attribute):
            return "enum" if self._ie_enum_class(node.value) is not None else None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in self.funcs:
            d = self._ie_ret_decl(self.funcs[node.func.id])
            return d.kind if d is not None else None
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
            return "int" if "int" in (self._ie_peek(node.left), self._ie_peek(node.right)) else None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return self._ie_peek(node.operand)
        return None

    # ================================================================ enums in expressions
    def _ie_enum_class(self, node):
        """The EnumInfo of a Name node naming an enum class (not shadowed), or None."""
        if isinstance(node, ast.Name) and node.id in getattr(self, "enums", {}) and self.var_type(node.id) is None \
                and self._comp_get(node.id) is None:
            return self.enums[node.id]
        return None

    def _ie_enum_refused(self, info, node):
        if info.refusal is not None:
            r = P.Refusal(node, "a use of the enum `%s` (refused at its class)" % info.name, "Fix the refusal at the class.")
            r.cascade = True
            raise r

    def _ie_attr(self, n):
        """`C.A` of an enum class C: the member; any other attribute of the class is refused."""
        info = self._ie_enum_class(n.value)
        if info is None:
            return None
        self._ie_enum_refused(info, n)
        a = n.attr
        if a in info.members:
            return "%s.%s" % (info.viv, a), enum_t(info.name), 100
        from classes import is_keyattr
        if is_keyattr(a):
            self.refuse(n, "looking up a member of the enum `%s` by name (`%s[...]`)" % (info.name, info.name),
                        "Write the member: %s.%s." % (info.name, info.members[0]))
        self.refuse(n, "the attribute `%s` of the enum class `%s`" % (a, info.name),
                    "py2viv translates the members (%s); the class's other attributes are not translated." % ", ".join(info.members))

    def _ie_member_attr(self, n, base, bt, bp):
        """`m.name` of a member; `.value` and the rest refused."""
        if n.attr == "name":
            return "str(%s)" % base, P.STR, 100
        if n.attr == "value":
            self.refuse(n, "the `.value` of an enum member", "py2viv does not carry the members' values; compare members, "
                        "or keep a def that maps a member to its value.")
        self.refuse(n, "the attribute `.%s` of an enum member" % n.attr, "A member has .name (and .value, not translated).")

    def _ie_class_value(self, n):
        info = self._ie_enum_class(n)
        if info is None:
            return
        self._ie_enum_refused(info, n)
        p = self.parents.get(n)
        if isinstance(p, ast.Call) and p.func is n:
            self.refuse(p, "looking up a member of the enum `%s` by value (`%s(...)`)" % (info.name, info.name),
                        "Write the member: %s.%s." % (info.name, info.members[0]))
        self.refuse(n, "the enum class `%s` used as a value (iterated, measured, passed)" % info.name,
                    "py2viv translates the members (%s.%s), comparisons, printing and .name." % (info.name, info.members[0]))

    def _ie_enum_text(self, node, code, t, conv):
        """str(m) of a member: Python's `C.A`."""
        if conv == "repr":
            self.refuse(node, "repr of an enum member", "Python writes <%s.A: value>; print the member, or its .name." % t.name)
        return "%s + str(%s)" % (P.viv_str(t.name + "."), code), P.STR, 60

    def _ie_print_enum(self, node, code, t, render, spec, parts, values):
        if render == "repr":
            self._ie_enum_text(node, code, t, "repr")
        if spec:
            self.refuse(node, "a format spec on an enum member", "Python's format of a member differs between versions; "
                        "format its text: f\"{str(m):>10}\".")
        parts.append({"kind": "str", "text": t.name + "."})
        self._print_value(node, "str(%s)" % code, P.STR, "str", None, parts, values)
        parts[-1]["sub"] = True  # Python records the member as an object: compared as the line's text (diff.py)

    # ================================================================ narrowing: if x is None
    def _ie_narrow_test(self, test):
        """(name, is_none) for `x is None` / `x is not None` of an optional variable, else None."""
        if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], (ast.Is, ast.IsNot))):
            return None
        a, b = test.left, test.comparators[0]
        if self._ie_is_none(a):
            a, b = b, a
        if not (isinstance(a, ast.Name) and self._ie_is_none(b)):
            return None
        if self._comp_get(a.id) is not None:
            return None
        t = self.var_type(a.id)
        if t is None or t.kind != "opt":
            return None
        return a.id, isinstance(test.ops[0], ast.Is)

    def _ie_narrow_stores(self, name, body):
        """Where x is known not to be None, a read of x is the value inside the optional (the
        match's binding); x = e stores Some(e) into the optional. Refused: a read of x after a
        store in that branch (it would read the old value), a store inside a loop there (the next
        pass would), and `del x`."""
        def pos(m):
            return (m.lineno, m.col_offset)
        names = [(m, par) for st in body for par in [None] for m in ast.walk(st) if isinstance(m, ast.Name) and m.id == name]
        stores = [m for m, _ in names if isinstance(m.ctx, (ast.Store, ast.Del))]
        if not stores:
            return
        for m in stores:
            if isinstance(m.ctx, ast.Del):
                self.refuse(m, "`del %s`" % name, "Viv frees nothing by hand; drop the `del`.")
        first = min(stores, key=pos)
        for m, _ in names:
            if isinstance(m.ctx, ast.Load) and pos(m) > pos(first):
                self.refuse(m, "`%s` read after it is assigned where it was known not to be None" % name,
                            "py2viv reads `%s` there as the value the optional held (a match binding); read the new "
                            "value from the name you computed it in." % name)
        for st in body:
            for loop in ast.walk(st):
                if isinstance(loop, (ast.For, ast.While)) and any(isinstance(m, ast.Name) and m.id == name and
                                                                 isinstance(m.ctx, ast.Store) for m in ast.walk(loop)):
                    self.refuse(loop, "`%s` assigned in a loop where it is known not to be None" % name,
                                "The loop's next pass would read the value the optional held before; assign it after the loop.")

    def _ie_narrow(self, s):
        """`if x is None: A else: B` (or `is not None`, B first): a match on x."""
        nt = self._ie_narrow_test(s.test)
        if nt is None:
            return False
        name, is_none = nt
        none_body, some_body = (s.body, s.orelse) if is_none else (s.orelse, s.body)
        self._ie_narrow_stores(name, some_body)
        t = self.var_type(name)
        binder = self.new_tmp(re.sub(r"\W", "", P.viv_name(name))[:12] or "v")
        self.flush_pre(s)
        self.emit("match %s {" % self.vname(name), s)
        self.indent += 1
        self.emit("None => {", s)
        self._body(none_body)
        self.emit("}", s)
        self.emit("Some(%s) => {" % binder, s)
        env = self._comp_push()
        env[name] = (binder, t.elts[0])
        narrowed = getattr(self, "_ie_narrowed", {})
        self._ie_narrowed = dict(narrowed, **{name: env})
        try:
            self._body(some_body)
        finally:
            self._comp_pop()
            self._ie_narrowed = narrowed
        self.emit("}", s)
        self.indent -= 1
        self.emit("}", s)
        return True

    def _ie_enum_stmt(self, s):
        """The class statement of an enum: `enum C { A, B }`."""
        info = self.enum_nodes[id(s)]
        if info.refusal is not None:
            self.refuse(*info.refusal)
        self._flush_comments(s.lineno)
        doc = ast.get_docstring(s)
        if doc:
            for dl in doc.strip().splitlines():
                self.emit(("# " + dl.strip()).rstrip(), s)
        self.emit("enum %s { %s }" % (info.viv, ", ".join(info.members)), s)
