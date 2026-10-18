# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""strs.py: Python's `str`, and lists of them, as Viv strings.

A mixin of the translator (py2viv.py). Viv's strings are Python's by construction (code points,
Python's split, strip, find, repr, float and int parsing, CPython 3.11's Unicode tables), so most
of the mapping is one Viv builtin per Python operation, and the rules for the methods and the
builtins are data (rules/strings.toml). What needs the translator's state is here:

  - literals (any text, escaped as the Viv lexer reads it), and the Unicode check: Viv's
    character data is CPython 3.11's (Unicode 14.0.0); under a Python whose Unicode is another,
    a literal beyond ASCII is refused and text read at run time (a file, chr) is checked to be
    ASCII, where the two agree;
  - the text of a value: str(x), repr(x), format(x, spec), an f-string, str.format and
    %-formatting, each written as Python writes it (an int's digits, a float's repr, True and
    False, a list of floats or ints or strings as Python prints it), or refused;
  - `+`, `*`, the comparisons, `in`, truth, indexing and slicing (Python's clamped slices and
    negative indices are Viv's own for strings and lists of strings);
  - `sep.join(...)` of a list, a string, a generator or a comprehension;
  - files: open(p).read(), Path(p).read_text(), `with open(p) as f` (read, iterate, readlines,
    write), `for line in open(p)`, Path(p).write_text(s).

A list of strings is immutable in Viv: py2viv grows one by `lst = append(lst, s)` where the list
is provably its own (the growable-list proof of py2viv.py, `grow_refusal`), and translates its
other methods by rebinding (`lst = lst[:k] + lst[k + 1:]`) under the same proof.
"""

import ast
import re
import string
import sys
import unicodedata

P = None  # py2viv's module, set by the translator (as funcs.P is)

# The Unicode data Viv's string builtins carry (lang/viv/src/interp/unicode_tables.rs, generated
# from CPython 3.11).
VIV_UNICODE = "14.0.0"
HOST_UNICODE = unicodedata.unidata_version

# The methods of str py2viv maps (rules/strings.toml); used to guess that a parameter of a def
# no call types is a string.
STR_ONLY_METHODS = {"upper", "lower", "strip", "lstrip", "rstrip", "split", "rsplit", "splitlines", "join", "find",
                    "rfind", "rindex", "replace", "startswith", "endswith", "isspace", "isalpha", "isalnum",
                    "isdigit", "isdecimal", "isnumeric", "isupper", "islower", "isascii", "isprintable", "ljust",
                    "rjust", "center", "zfill", "casefold", "title", "capitalize", "swapcase", "partition",
                    "rpartition", "encode", "format", "expandtabs", "translate", "removeprefix", "removesuffix"}

# Python's format spec: [[fill]align][sign][z][#][0][width][grouping][.precision][type]
SPEC_RE = re.compile(r"(?:(?P<fill>.)?(?P<align>[<>=^]))?(?P<sign>[-+ ])?(?P<z>z)?(?P<alt>#)?(?P<zero>0)?"
                     r"(?P<width>\d+)?(?P<group>[,_])?(?:\.(?P<prec>\d+))?(?P<type>[bcdeEfFgGnosxX%])?", re.S)

# printf-style: %[(name)][flags][width][.precision][length]type
PCT_RE = re.compile(r"%(?:\((?P<key>[^)]*)\))?(?P<flags>[-+ #0]*)(?P<width>\*|\d+)?(?:\.(?P<prec>\*|\d*))?[hlL]?"
                    r"(?P<type>[diouxXeEfFgGcrsa%])")

MAX_WIDTH = 100  # Viv's fmt: a width is at most 100, a precision at most 40
MAX_PREC = 40


def parse_spec(spec):
    m = SPEC_RE.fullmatch(spec)
    return m.groupdict() if m else None


class StringsMixin:
    """The translator's methods for strings (see the module docstring)."""

    # ------------------------------------------------ types
    def _is_text_type(self, t):
        return t is not None and t.kind in ("str", "strs")

    # ------------------------------------------------ literals
    def _str_lit(self, text, node):
        """A Python str constant as a Viv literal."""
        if HOST_UNICODE != VIV_UNICODE and any(ord(ch) > 127 for ch in text):
            self.refuse(node, "a string literal beyond ASCII under a Python whose Unicode data is %s" % HOST_UNICODE,
                        "Viv's character data is CPython 3.11's (Unicode %s); translate with Python 3.11, or keep the "
                        "text ASCII." % VIV_UNICODE)
        try:
            return P.viv_str(text)
        except ValueError:
            self.refuse(node, "a string with a lone surrogate", "A Viv string is UTF-8 text, which cannot hold one.")

    def _host_ascii(self, code):
        """Text a program reads at run time (a file, chr of a computed number), checked to be ASCII
        where the host Python's Unicode data is not Viv's (the two agree on ASCII)."""
        if HOST_UNICODE == VIV_UNICODE:
            return code
        return "%s(%s)" % (self.use("py_ascii_text"), code)

    # ------------------------------------------------ the text of a value
    def _str_of(self, node, code, t, prec=100, conv="str"):
        """str(x) (conv "str") or repr(x) (conv "repr") of a translated value, as Python writes
        it: (code, STR, prec)."""
        k = t.kind
        if k == "int":
            k = "num"  # a Viv Int: its digits, as Python writes an int (ints.py)
        if k == "enum":
            return self._ie_enum_text(node, code, t, conv)  # `C.A` (ints.py)
        if k == "str":
            if conv == "repr":
                return "repr(%s)" % code, P.STR, 100
            return code, P.STR, prec
        if k == "strs":
            return "str(%s)" % code, P.STR, 100
        if k == "num":
            if t.py == "int":
                return "fmt(%s, \"d\")" % P.Translator._intfix(code, t), P.STR, 100
            if t.py == "mixed":
                self.refuse(node, "%s of a number that is an int on some paths and a float on others" % conv,
                            "Python writes the two differently (3 or 3.0); convert it with float(...) or int(...) first.")
            # Python's str of a float is its repr (3.0); Viv's str writes a whole
            # number as its digits, so the text is fmt(x)
            return "fmt(%s)" % code, P.STR, 100
        if k == "bool":
            return "%s(%s)" % (self.use("py_bstr"), code), P.STR, 100
        if k == "arr" and t.container == "list" and t.rank in (1, 2) and not t.elts and t.py in ("int", "float") \
                and t.name != "ndrows":
            if t.py == "int":
                return "fmt(%s, \"d\")" % code, P.STR, 100
            return "fmt(%s)" % code, P.STR, 100
        if k == "unknown":
            r = P.Refusal(node, "the text of a value of unknown type (it comes from a construct refused elsewhere)",
                          "Fix the refusal that produced it.")
            r.cascade = True
            raise r
        what = {"arr": "a NumPy array" if t.container == "nd" else "a list of Bools, or of ints beside floats",
                "carr": "a complex array", "cnum": "a complex number", "none": "None"}.get(k, self._kind_words(t))
        if k == "arr" and t.container == "list" and t.rank > 2:
            what = "a list of rank %d" % t.rank
        self.refuse(node, "%s of %s" % (conv, what),
                    "py2viv writes the text of strings, ints, floats, Bools, and lists of these; NumPy's own layout "
                    "(and a complex number's) is not reproduced: print it, or format its entries.")

    def _str_format(self, node, code, t, spec, conv=None, prec=100):
        """format(x, spec), the field `{x!conv:spec}` of an f-string: (code, STR, prec)."""
        if conv in ("r", "repr"):
            code, t, prec = self._str_of(node, code, t, prec, "repr")
        elif conv in ("s", "str"):
            code, t, prec = self._str_of(node, code, t, prec, "str")
        elif conv == "a":
            self.refuse(node, "the conversion `!a`", "Python escapes every character beyond ASCII; use !r for ASCII text.")
        if spec == "":
            return self._str_of(node, code, t, prec, "str")
        d = parse_spec(spec)
        if d is None:
            self.refuse(node, "the format spec `%s`" % spec, "Python raises ValueError (an invalid format specifier).")
        if t.kind == "str":
            return self._str_spec(node, code, d, spec), P.STR, 100
        if t.kind in ("num", "int") or t.kind == "bool":
            return self._num_spec(node, code, t, d, spec), P.STR, 100
        if t.kind == "unknown":
            self._str_of(node, code, t, prec)
        self.refuse(node, "the format spec `%s` on %s" % (spec, self._kind_words(t)),
                    "Python's format of a list, an array or an object takes no spec but an empty one (TypeError); "
                    "format the entries.")

    def _str_spec(self, node, code, d, spec):
        """format(s, spec) of a string: [[fill]align][width][.precision][s], Viv's fmt(s, spec)."""
        if d["align"] == "=":
            self.refuse(node, "the format spec `%s` on a string" % spec, "Python raises ValueError ('=' alignment not allowed in string format specifier).")
        if d["sign"] or d["z"] or d["alt"] or d["group"] or d["zero"]:
            self.refuse(node, "the format spec `%s` on a string" % spec,
                        "Python raises ValueError for a sign, `z`, `#` or grouping on a string; `0` is refused (its fill "
                        "depends on the alignment).")
        if d["type"] not in (None, "s"):
            self.refuse(node, "the format spec `%s` on a string" % spec,
                        "Python raises ValueError (Unknown format code '%s' for object of type 'str')." % d["type"])
        width = int(d["width"]) if d["width"] else 0
        if width > MAX_WIDTH:
            self.refuse(node, "a width of %d in a format spec" % width, "Viv's fmt takes widths up to %d." % MAX_WIDTH)
        vspec = "%s%s%s%s" % (d["fill"] or "", d["align"] or "", d["width"] or "",
                              "." + d["prec"] if d["prec"] is not None else "")
        if not vspec:
            return code
        return "fmt(%s, %s)" % (code, self._str_lit(vspec, node))

    def _num_spec(self, node, code, t, d, spec):
        """format(x, spec) of a number: Viv's fmt(x, "[width][.precision]type") for Python's
        [width][.precision][type] with type d, e, f or g (an int's digits for an empty type),
        then fmt(text, "[fill]align width") for an alignment. Everything else is refused."""
        if t.kind == "bool":
            self.refuse(node, "the format spec `%s` on a Bool" % spec, "Python formats a Bool as the int 1 or 0 under a spec; convert it first.")
        if d["sign"] or d["z"] or d["alt"] or d["group"] or d["zero"] or d["align"] == "=":
            self.refuse(node, "the format spec `%s`" % spec,
                        "Viv's fmt takes [[fill]align][width][.precision] with type d, e, f or g: no sign, `z`, `#`, `0` "
                        "or grouping.", gap="R4")
        typ = d["type"] or ""
        if typ not in ("", "d", "e", "f", "g"):
            self.refuse(node, "the format spec `%s`" % spec,
                        "Viv's fmt takes the types d, e, f and g (no upper case, `n`, `%`, or the int bases).", gap="R4")
        if t.py == "mixed" and typ not in ("e", "f", "g"):
            self.refuse(node, "the format spec `%s` on a number that is an int on some paths and a float on others" % spec,
                        "Python writes an int and a float differently under this spec; convert it with float(...) first.")
        is_int = t.py == "int"
        # a float with no type is its str (repr) in the width: Viv's spec with
        # no type writes a whole number without `.0`, so the repr is padded
        # as text, right-aligned as Python aligns a number
        if typ == "" and not is_int and d["prec"] is None:
            width = int(d["width"]) if d["width"] else 0
            if width > MAX_WIDTH:
                self.refuse(node, "a width of %d in a format spec" % width, "Viv's fmt takes widths up to %d." % MAX_WIDTH)
            inner = "fmt(%s)" % code
            if not width:
                return inner
            return "fmt(%s, %s)" % (inner, self._str_lit((d["fill"] or "") + (d["align"] or ">") + str(width), node))
        num = (("." + d["prec"]) if d["prec"] is not None else "") + typ
        if d["align"] is None and d["fill"] is None:
            vspec = P.viv_spec((d["width"] or "") + num, is_int)
            if vspec is None:
                self.refuse(node, "the format spec `%s`" % spec,
                            "Python raises (a precision on an int, `d` on a float) or its rule differs from Viv's fmt "
                            "(a precision with no type).", gap="R4")
            return "fmt(%s, %s)" % (P.Translator._intfix(code, t), P.viv_str(vspec))
        vnum = P.viv_spec(num, is_int)
        if vnum is None:
            self.refuse(node, "the format spec `%s`" % spec,
                        "Python raises (a precision on an int, `d` on a float) or its rule differs from Viv's fmt "
                        "(a precision with no type).", gap="R4")
        inner = "fmt(%s, %s)" % (P.Translator._intfix(code, t), P.viv_str(vnum))
        width = int(d["width"]) if d["width"] else 0
        if width > MAX_WIDTH:
            self.refuse(node, "a width of %d in a format spec" % width, "Viv's fmt takes widths up to %d." % MAX_WIDTH)
        if not width:
            return inner
        return "fmt(%s, %s)" % (inner, self._str_lit((d["fill"] or "") + d["align"] + str(width), node))

    # ------------------------------------------------ f-strings, str.format, %-formatting
    def _concat(self, pieces):
        """The concatenation of string codes [(code, prec)], or the empty string."""
        pieces = [(c, p) for c, p in pieces if c != '""']
        if not pieces:
            return '""', P.STR, 100
        if len(pieces) == 1:
            return pieces[0][0], P.STR, pieces[0][1]
        out = []
        for k, (c, p) in enumerate(pieces):
            out.append(self._paren(c, p, 60 if k == 0 else 61))
        return " + ".join(out), P.STR, 60

    def _e_JoinedStr(self, n):
        """An f-string as a value: its text and the text of each field, concatenated."""
        pieces = []
        for v in n.values:
            if isinstance(v, ast.Constant):
                pieces.append((self._str_lit(v.value, n), 100))
                continue
            pieces.append(self._fstr_field(v, n))
        return self._concat(pieces)

    def _field_spec(self, v):
        if v.format_spec is None:
            return ""
        if not all(isinstance(x, ast.Constant) for x in v.format_spec.values):
            self.refuse(v.format_spec, "a computed format spec", "Write the spec literally.")
        return "".join(x.value for x in v.format_spec.values)

    def _fstr_field(self, v, n):
        spec = self._field_spec(v)
        conv = {-1: None, 115: "s", 114: "r", 97: "a"}[v.conversion]
        node = v.value
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return self._str_lit(self._const_field(node.value, conv, spec, node), n), 100
        code, t, prec = self.expr(node, keep=True)  # an Int, a member (ints.py)
        c, _, p = self._str_format(node, code, t, spec, conv, prec)
        return c, p

    def _const_field(self, text, conv, spec, node):
        """The text of a field whose value is a string constant, as Python formats it."""
        try:
            val = {"r": repr(text), "a": ascii(text), "s": text, None: text}[conv]
            return format(val, spec)
        except (ValueError, TypeError) as e:
            self.refuse(node, "the format spec `%s` on a string" % spec, "Python raises %s (%s)." % (type(e).__name__, e))

    def _str_method_format(self, n, recv, args, kws):
        """`"...{}...".format(a, b, k=c)` with a literal template: the pieces of an f-string.
        Each argument is evaluated once, in order, as Python evaluates them; one with an effect
        (it calls a def of the program, or draws) must be used exactly once, in its place."""
        if not (isinstance(recv, ast.Constant) and isinstance(recv.value, str)):
            self.refuse(n, "str.format of a template that is not a literal", "Write the template in the call, or an f-string.")
        try:
            parsed = list(string.Formatter().parse(recv.value))
        except ValueError as e:
            self.refuse(n, "this str.format template", "Python raises ValueError (%s)." % e)
        for a in args:
            if isinstance(a, ast.Starred):
                self.refuse(a, "a starred argument of str.format", "Pass the values one by one.")
        for k in kws:
            if k.arg is None:
                self.refuse(k.value, "`**kwargs` in str.format", "Pass the values by name.")
        order = []  # which argument each field reads, in the template's order
        auto = None
        for lit, field, spec, conv in parsed:
            if field is None:
                continue
            if spec and "{" in spec:
                self.refuse(n, "a nested field in a format spec", "Write the spec literally.")
            if field == "":
                if auto is False:
                    self.refuse(n, "automatic and manual field numbering in one template", "Python raises ValueError.")
                auto = True
                key = sum(1 for o in order if isinstance(o[0], int))
            elif field.isdigit():
                if auto is True:
                    self.refuse(n, "automatic and manual field numbering in one template", "Python raises ValueError.")
                auto = False
                key = int(field)
            elif field.isidentifier():
                key = field
            else:
                self.refuse(n, "the field `{%s}` (an attribute or an item)" % field, "Read it into a variable first.")
            order.append((key, spec, conv))
        kwmap = {k.arg: k.value for k in kws}
        for key, _, _ in order:
            if isinstance(key, int) and key >= len(args):
                self.refuse(n, "str.format with too few arguments", "Python raises IndexError.")
            if isinstance(key, str) and key not in kwmap:
                self.refuse(n, "str.format without the argument `%s`" % key, "Python raises KeyError.")
        # evaluate every argument once, in order
        nodes = list(args) + [k.value for k in kws]
        keys = list(range(len(args))) + [k.arg for k in kws]
        vals = {}
        effect_keys = []
        for key, a in zip(keys, nodes):
            code, t, prec = self.expr(a)
            if self._code_has_effect(code):
                effect_keys.append(key)
            vals[key] = (a, code, t, prec)
        used = [o[0] for o in order]
        for key in effect_keys:
            if used.count(key) != 1:
                self.refuse(vals[key][0], "a str.format argument that calls a def, used %d times in the template" % used.count(key),
                            "Python evaluates it once; assign it to a variable first.")
        if [k for k in used if k in effect_keys] != effect_keys:
            self.refuse(n, "str.format arguments that call defs, read in another order than written",
                        "Python evaluates the arguments in the order written; assign them to variables first.")
        pieces = []
        oi = 0
        for lit, field, spec, conv in parsed:
            if lit:
                pieces.append((self._str_lit(lit, n), 100))
            if field is None:
                continue
            key, spec, conv = order[oi]
            oi += 1
            a, code, t, prec = vals[key]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                pieces.append((self._str_lit(self._const_field(a.value, conv, spec or "", a), a), 100))
                continue
            c, _, p = self._str_format(a, code, t, spec or "", conv, prec)
            pieces.append((c, p))
        return self._concat(pieces)

    def _code_has_effect(self, code):
        if P.rule_engine.DRAWS.search(code):
            return True
        return any(re.search(r"(?<![\w.])%s\(" % re.escape(P.viv_name(f)), code) for f in self._effect_names())

    def _str_percent(self, n):
        """`"..." % x`, `"..." % (a, b)`: printf-style formatting of a literal template, for
        %s %r %d %i %f %e %g %% with a width, a precision and the flag `-`. Anything else
        (a mapping key, `*`, other flags and types) is refused."""
        tmpl = n.left.value
        right = n.right
        items = list(right.elts) if isinstance(right, ast.Tuple) else [right]
        convs = []
        pos = 0
        pieces_lit = []
        for m in PCT_RE.finditer(tmpl):
            pieces_lit.append(tmpl[pos:m.start()])
            pos = m.end()
            convs.append(m)
        tail = tmpl[pos:]
        if "%" in "".join(pieces_lit) + tail:
            self.refuse(n, "this %-format template", "Python raises ValueError (an incomplete or unsupported format).")
        for m in convs:
            if m.group("key") is not None:
                self.refuse(n, "%-formatting with a mapping key", "Use an f-string.")
            if m.group("width") == "*" or m.group("prec") == "*":
                self.refuse(n, "%-formatting with `*`", "Write the width and precision literally.")
            if set(m.group("flags")) - {"-"}:
                self.refuse(n, "%%-formatting with the flags `%s`" % m.group("flags"), "Only `-` is translated; use an f-string.")
            if m.group("type") not in "sdifeg%r":
                self.refuse(n, "%%-formatting with `%%%s`" % m.group("type"), "Only %s %r %d %i %f %e %g and %% are translated.")
        need = sum(1 for m in convs if m.group("type") != "%")
        if not isinstance(right, ast.Tuple):
            _, rt, _ = self._expr_peek(right)
            if rt.kind == "tuple" or rt.kind == "arr" and rt.name == "tuple" or rt.kind == "map" or rt.kind == "rec":
                self.refuse(right, "%-formatting of a value held in a variable that is a tuple or a mapping",
                            "Write the tuple in the expression: \"...\" % (a, b).")
        if need != len(items):
            self.refuse(n, "%%-formatting with %d values for %d fields" % (len(items), need), "Python raises TypeError.")
        pieces = []
        k = 0
        for lit, m in zip(pieces_lit, convs):
            if lit:
                pieces.append((self._str_lit(lit, n), 100))
            typ = m.group("type")
            if typ == "%":
                pieces.append(('"%"', 100))
                continue
            a = items[k]
            k += 1
            code, t, prec = self.expr(a)
            left = "-" in m.group("flags")
            width = m.group("width")
            p = m.group("prec")
            if typ in "sr":
                c, _, cp = self._str_of(a, code, t, prec, "repr" if typ == "r" else "str")
                spec = ("<" if left else ">") + width if width else ""
                if p is not None:
                    spec += "." + (p or "0")
                if spec:
                    c = "fmt(%s, %s)" % (c, self._str_lit(spec, n))
                    cp = 100
                pieces.append((c, cp))
                continue
            if t.kind == "bool":
                code, t = "%s(%s)" % (self.use("py_b01"), code), P.NUM_I
            if t.kind != "num":
                self.refuse(a, "%%%s of %s" % (typ, self._kind_words(t)), "Python raises TypeError.")
            if typ in "di":
                if p is not None:
                    self.refuse(a, "%-formatting of an int with a precision", "Python pads the digits with zeros; use an f-string.")
                if t.py != "int":
                    code, t = "%s(%s)" % (self.use("py_trunc"), code), P.NUM_I  # %d of a float truncates, as int(x)
                inner = "fmt(%s, \"d\")" % P.Translator._intfix(code, t)
            else:
                if t.py == "mixed" or t.py == "int":
                    code = "%s + 0.0" % self._paren(code, prec, 61) if t.py == "int" else code
                inner = "fmt(%s, %s)" % (code, P.viv_str("." + (p if p else ("0" if p == "" else "6")) + typ))
                if int(p or 6) > MAX_PREC:
                    self.refuse(a, "a precision above %d" % MAX_PREC, "Viv's fmt takes precisions up to %d." % MAX_PREC)
            if width:
                if int(width) > MAX_WIDTH:
                    self.refuse(a, "a width of %s" % width, "Viv's fmt takes widths up to %d." % MAX_WIDTH)
                inner = "fmt(%s, %s)" % (inner, self._str_lit(("<" if left else ">") + width, n))
            pieces.append((inner, 100))
        if tail:
            pieces.append((self._str_lit(tail, n), 100))
        return self._concat(pieces)

    def _expr_peek(self, node):
        """Translate node now and keep the result for the next self.expr(node) of this statement."""
        r = self.expr(node)
        self._xmemo[id(node)] = r
        return r

    def _stop_text(self, node):
        """The text of an exception's message computed at run time (`raise E(f"...")`, a string
        variable, a number), as Python's str(e) writes it; None where it cannot be written (the
        translation then keeps the type and the literal parts, compared by type)."""
        memo = self._xmemo  # reset at every statement
        if ("stop", id(node)) in memo:
            return memo[("stop", id(node))]
        saved_pre, saved_tmp = list(self.pre), self.tmp
        try:
            code, t, prec = self.expr(node)
            code = self._str_of(node, code, t, prec, "str")[0]
        except P.Refusal:
            self.pre, self.tmp = saved_pre, saved_tmp
            code = None
        memo[("stop", id(node))] = code
        return code

    # ------------------------------------------------ operators
    def _str_binop(self, n, op, a, ta, pa, b, tb, pb):
        """`+`, `*`, `%` with a string or a list of strings on either side; None when neither is one."""
        if not (self._is_text_type(ta) or self._is_text_type(tb)):
            return None
        sym = self._opsym(op)
        if isinstance(op, ast.Add):
            if ta.kind == tb.kind and ta.kind in ("str", "strs"):
                return "%s + %s" % (self._paren(a, pa, 60), self._paren(b, pb, 61)), ta, 60
            if ta.kind == "unknown" or tb.kind == "unknown":
                return "%s + %s" % (self._paren(a, pa, 60), self._paren(b, pb, 61)), ta if ta.kind != "unknown" else tb, 60
            self.refuse(n, "`+` of %s and %s" % (self._kind_words(ta), self._kind_words(tb)),
                        "Python raises TypeError; convert with str(...), or put the string in a list.")
        if isinstance(op, ast.Mult):
            tk, kn = (tb, n.right) if ta.kind == "str" else (ta, n.left)
            if ta.kind == "strs" or tb.kind == "strs":
                self.refuse(n, "`*` of a list of strings (repetition)", "Build it with a loop, or `+`.")
            if tk.kind == "num" and tk.py == "int":
                # s * n or n * s, in the order written (Viv takes both, n <= 0 giving "")
                return "%s * %s" % (self._paren(a, pa, 70), self._paren(b, pb, 71)), P.STR, 70
            if tk.kind == "bool":
                self.refuse(kn, "a string repeated a Bool number of times", "Python repeats it 0 or 1 times; write `s if c else \"\"`.")
            self.refuse(n, "`*` of a string and %s" % self._kind_words(tk),
                        "Python repeats a string an int number of times (TypeError for anything else).")
        self.refuse(n, "`%s` with %s" % (sym, self._kind_words(ta if self._is_text_type(ta) else tb)),
                    "Python raises TypeError (or formats, for `%` of a literal template on the left).")

    def _str_compare(self, n, sym, lc, lt, lp, rc, rt, rp):
        """A comparison with a string or a list of strings on a side: Python's order by code point,
        `==` of two lists; None when neither side is text."""
        if not (self._is_text_type(lt) or self._is_text_type(rt)):
            return None
        if lt.kind == "unknown" or rt.kind == "unknown":
            return "%s %s %s" % (self._paren(lc, lp, 51), sym, self._paren(rc, rp, 51))
        if lt.kind != rt.kind:
            other = rt if self._is_text_type(lt) else lt
            if sym in ("==", "!=") and other.kind in ("num", "bool", "none") and \
                    not self._code_has_effect(lc) and not self._code_has_effect(rc):
                # a string is never equal to a number, a Bool or None: Python's answer, whatever the values
                return "false" if sym == "==" else "true"
            self.refuse(n, "comparing %s with %s" % (self._kind_words(lt), self._kind_words(rt)),
                        "Python gives False for ==, True for != and raises TypeError for an order; compare values of one kind.")
        if lt.kind == "strs" and sym not in ("==", "!="):
            self.refuse(n, "`%s` of two lists of strings" % sym, "Python compares them item by item; compare the items.")
        return "%s %s %s" % (self._paren(lc, lp, 51), sym, self._paren(rc, rp, 51))

    def _str_in(self, n, op, lc, lt, lp, right):
        """`x in s`, `x in lst`, not in: substring and membership; None when not text."""
        rc, rt, rp = self._expr_peek(right)
        if not (self._is_text_type(rt) or self._is_text_type(lt) and rt.kind in ("str", "strs", "unknown")):
            return None
        self._xmemo.pop(id(right), None)
        if rt.kind == "unknown":
            code = "%s in %s" % (self._paren(lc, lp, 51), self._paren(rc, rp, 51))
        elif lt.kind != "str":
            self.refuse(n, "`in` of %s in %s" % (self._kind_words(lt), self._kind_words(rt)),
                        "Python raises TypeError for a string (and finds nothing in a list of strings); look for a string.")
        else:
            code = "%s in %s" % (self._paren(lc, lp, 51), self._paren(rc, rp, 51))
        return ("not (%s)" % code) if isinstance(op, ast.NotIn) else code

    def _str_truth(self, code, t):
        return "len(%s) > 0" % code

    # ------------------------------------------------ indexing and slicing
    def _str_subscript(self, n):
        """s[i], s[a:b:k], lst[i][j] of a string or a list of strings named or written in place:
        Python's positions (negative from the end, slice bounds clamped), which are Viv's for
        text. None when the innermost base is not one of these (a base computed another way is
        translated by the subscript's own path, which hands a text base to _str_chain)."""
        levels = self._levels(n)
        inner = levels[0][0].value
        if isinstance(inner, ast.Name):
            c = self._comp_get(inner.id)
            t = c[1] if c is not None else self.var_type(inner.id)
            if not self._is_text_type(t):
                return None
        elif not (isinstance(inner, ast.Constant) and isinstance(inner.value, str) or isinstance(inner, ast.JoinedStr)):
            return None
        return self._str_chain(n, self.expr(inner))

    def _str_chain(self, n, base):
        """The subscripts of n, one level at a time, applied to its innermost base, a string or a
        list of strings translated as `base` (code, T, prec)."""
        levels = self._levels(n)
        code, t, prec = base
        for lv, items in levels:
            if len(items) != 1 or isinstance(lv.slice, ast.Tuple):
                self.refuse(lv, "a subscript of %s with several items" % self._kind_words(t),
                            "Python raises TypeError (a string and a list take one index or slice).")
            if not self._is_text_type(t):
                self.refuse(lv, "a subscript of %s" % self._kind_words(t), "Index the string or the list.")
            it = items[0]
            base = self._paren(code, prec, 100)
            if isinstance(it, ast.Slice):
                parts = []
                for b in (it.lower, it.upper, it.step):
                    if b is None or isinstance(b, ast.Constant) and b.value is None:
                        parts.append("")
                        continue
                    parts.append(self._str_index(b, "a slice bound"))
                if it.step is None:
                    parts = parts[:2]
                code = "%s[%s]" % (base, ":".join(parts))
            else:
                code = "%s[%s]" % (base, self._str_index(it, "an index"))
                t = P.STR
            prec = 100
        return code, t, 100

    def _str_index(self, node, what):
        c, t, _ = self.expr(node)
        if t.kind == "num" and t.py == "int" or t.kind == "unknown":
            return c
        if t.kind == "bool":
            self.refuse(node, "%s that is a Bool" % what, "Python reads True as 1; write the int.")
        self.refuse(node, "%s that is %s" % (what, "a float" if t.kind == "num" and t.py == "float" else
                                             "a number that is an int on some paths and a float on others" if t.kind == "num"
                                             else self._kind_words(t)),
                    "Python takes only ints (TypeError otherwise); convert with int(...).")

    # ------------------------------------------------ join
    def _str_join(self, node, sep, x):
        """sep.join(x) (the rule str.join): x a list of strings, a string (its characters), a
        generator or a comprehension of strings (built as a list first)."""
        sc = sep.code
        xn = x.node
        if isinstance(xn, (ast.List, ast.Tuple)) and not xn.elts:
            return '""', P.STR, 100  # nothing to join
        if isinstance(xn, ast.Tuple):
            xn = ast.copy_location(ast.List(elts=xn.elts, ctx=ast.Load()), xn)  # a tuple of strings, read once
            xc, xt, _ = self.expr(xn)
        elif isinstance(xn, ast.GeneratorExp):
            xn = ast.copy_location(ast.ListComp(elt=xn.elt, generators=xn.generators), xn)
            ast.fix_missing_locations(xn)
            xc, xt, _ = self.expr(xn)
        else:
            xc, xt = x.code, x.t
        if xt.kind == "strs":
            return "join(%s, %s)" % (sc, xc), P.STR, 100
        if xt.kind == "str":
            return "join(%s, %s(%s))" % (sc, self.use("py_chars"), xc), P.STR, 100
        if xt.kind == "unknown":
            r = P.Refusal(x.node, "join of a value of unknown type (it comes from a construct refused elsewhere)",
                          "Fix the refusal that produced it.")
            r.cascade = True
            raise r
        self.refuse(x.node, "join of %s" % self._kind_words(xt),
                    "Python joins strings only (TypeError for numbers): join str(x) of each, "
                    "`\", \".join(str(x) for x in xs)`.")

    # ------------------------------------------------ sorted
    def _str_set_of(self, it):
        """`set(x)` of a string or a list of strings: x's translation, kept for the next
        self.expr(x); None when `it` is not such a call. A set of anything else is refused
        here, since the only exact uses of a set (sorted, len) need Viv's `unique`."""
        if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "set" and self._no_def("set")):
            return None
        if len(it.args) != 1 or it.keywords or isinstance(it.args[0], ast.Starred):
            self.refuse(it, "set with these arguments", "Pass one string or list of strings: sorted(set(s)) is Viv's unique(s).")
        r = self._expr_peek(it.args[0])
        if not self._is_text_type(r[1]):
            self.refuse(it, "a set of %s" % self._kind_words(r[1]),
                        "Viv's unique gives the distinct characters of a string or strings of a list, sorted; the distinct "
                        "numbers of an array are lib/algo's unique (tier 1), not built yet.")
        self._xmemo.pop(id(it.args[0]), None)
        return r

    def _str_set_len(self, n):
        """len(set(x)) of text: len(unique(x)). None for any other len."""
        if len(n.args) != 1 or n.keywords:
            return None
        r = self._str_set_of(n.args[0])
        if r is None:
            return None
        return "len(unique(%s))" % r[0], P.NUM_I, 100

    def _str_sorted(self, n):
        """sorted(x) of a list of strings or of a string's characters, by code point (Python's
        order), stably; reverse=True as the sorted list read backward (equal strings cannot be
        told apart, so this is Python's reverse sort). sorted(set(x)) of text is Viv's
        unique(x). None when x is not text."""
        if len(n.args) != 1 or isinstance(n.args[0], ast.Starred):
            return None
        it = n.args[0]
        uniq = self._str_set_of(it)
        if uniq is not None:
            r = uniq
        elif isinstance(it, ast.GeneratorExp):
            lc = ast.copy_location(ast.ListComp(elt=it.elt, generators=it.generators), it)
            ast.fix_missing_locations(lc)
            saved_pre, saved_tmp = list(self.pre), self.tmp
            try:
                r = self.expr(lc)
            except P.Refusal:
                self.pre, self.tmp = saved_pre, saved_tmp
                return None
            if r[1].kind != "strs":
                self.pre, self.tmp = saved_pre, saved_tmp
                return None
        else:
            r = self._expr_peek(it)
            if not self._is_text_type(r[1]):
                return None
            self._xmemo.pop(id(it), None)
        code, t, _ = r
        call = "unique" if uniq is not None else "sorted"
        rev = False
        for k in n.keywords:
            if k.arg == "reverse" and isinstance(k.value, ast.Constant) and isinstance(k.value.value, bool):
                rev = k.value.value
            elif k.arg == "key" and isinstance(k.value, ast.Constant) and k.value.value is None:
                pass
            elif k.arg == "key":
                self.refuse(k.value, "sorted of strings with key=", "Sort by the strings themselves, or build (key, index) "
                            "pairs of numbers and sort those.")
            else:
                self.refuse(k.value, "sorted's `%s=`" % k.arg, "Write reverse=True or reverse=False.")
        return ("%s(%s)[::-1]" if rev else "%s(%s)") % (call, code), P.STRS, 100

    # ------------------------------------------------ lists of strings
    def _strs_method(self, ctx, m, views):
        """The list methods of a list of strings py2viv owns (a.grows): each rebinds the name."""
        node = ctx.node
        a = views[0]
        lst = a.node
        n = self.vname(lst.id)
        stmt = ctx.rule.context == "stmt"
        if m == "pop":
            i = views[1]
            ic = "-1"
            if i is not None and i.given:
                if i.t.kind != "num" or i.t.py != "int":
                    self.refuse(i.node, "list.pop with an index that is not an int", "Python raises TypeError; pass an int.")
                ic = i.code
            k = self.new_tmp("k")
            self.pre.append("%s = %s(len(%s), %s)" % (k, self.use("py_popk"), n, ic))
            drop = "%s = %s[:%s] + %s[%s + 1:]" % (n, n, k, n, k)
            if stmt:
                return drop, None, 100
            x = self.new_tmp("x")
            self.pre.append("%s = %s[%s]" % (x, n, k))
            self.pre.append(drop)
            return x, P.STR, 100
        if m == "clear":
            return "%s = %s[:0]" % (n, n), None, 100
        if m == "reverse":
            return "%s = %s[::-1]" % (n, n), None, 100
        if m == "sort":
            return "%s = sorted(%s)" % (n, n), None, 100
        if m == "insert":
            i, x = views[1], views[2]
            if i.t.kind != "num" or i.t.py != "int":
                self.refuse(i.node, "list.insert at an index that is not an int", "Python raises TypeError; pass an int.")
            if x.t.kind != "str":
                self.refuse(x.node, "inserting %s into a list of strings" % self._kind_words(x.t), "Python keeps it; keep one kind per list.")
            k = self.new_tmp("k")
            self.pre.append("%s = %s(len(%s), %s)" % (k, self.use("py_insk"), n, i.code))
            return "%s = %s[:%s] + [%s] + %s[%s:]" % (n, n, k, x.code, n, k), None, 100
        if m == "remove":
            x = views[2]
            if x.t.kind != "str":
                self.refuse(x.node, "removing %s from a list of strings" % self._kind_words(x.t), "Look for a string.")
            k = self.new_tmp("k")
            xc = x.code
            if not re.fullmatch(r"[A-Za-z_]\w*|\"[^\"\\]*\"", xc):
                t = self.new_tmp("x")
                self.pre.append("%s = %s" % (t, xc))
                xc = t
            self.pre.append("if not (%s in %s) {" % (xc, n))
            self.pre.append("    " + self._host_stop("remove_missing", node))
            self.pre.append("}")
            self.pre.append("%s = index(%s, %s)" % (k, n, xc))
            return "%s = %s[:%s] + %s[%s + 1:]" % (n, n, k, n, k), None, 100
        if m == "extend":
            x = views[2]
            xn = x.node
            if isinstance(xn, ast.GeneratorExp):
                xn = ast.copy_location(ast.ListComp(elt=xn.elt, generators=xn.generators), xn)
                ast.fix_missing_locations(xn)
                xc, xt, _ = self.expr(xn)
            else:
                xc, xt = x.code, x.t
            if xt.kind == "str":
                xc, xt = "%s(%s)" % (self.use("py_chars"), xc), P.STRS
            if xt.kind != "strs":
                self.refuse(x.node, "extending a list of strings by %s" % self._kind_words(xt), "Extend it by strings.")
            return "%s = %s + %s" % (n, n, xc), None, 100
        self.refuse(node, "list.%s of a list of strings" % m, "Not translated.")

    def _strs_index_count(self, ctx, m, a, x):
        if x.t.kind != "str":
            self.refuse(x.node, "`.%s` of %s in a list of strings" % (m, self._kind_words(x.t)),
                        "Python finds nothing (count 0, or ValueError for index); look for a string.")
        if m == "count":
            return "count(%s, %s)" % (a.code, x.code), P.NUM_I, 100
        xc = x.code
        if not re.fullmatch(r"[A-Za-z_]\w*|\"[^\"\\]*\"", xc):
            t = self.new_tmp("x")
            self.pre.append("%s = %s" % (t, xc))
            xc = t
        ac = a.code
        if not re.fullmatch(r"[A-Za-z_]\w*", ac):
            t = self.new_tmp("s")
            self.pre.append("%s = %s" % (t, ac))
            ac = t
        # Python's message names the item in repr: "'x' is not in list"
        self.pre.append("if not (%s in %s) {" % (xc, ac))
        self.pre.append("    stop(\"ValueError: \" + repr(%s) + \" is not in list\")" % xc)
        self.pre.append("}")
        self.stops.append({"line": ctx.node.lineno, "message": "ValueError: ... is not in list", "exact": False})
        return "index(%s, %s)" % (ac, xc), P.NUM_I, 100

    # ------------------------------------------------ files
    OPEN_READ = ("r", "rt", "tr")
    OPEN_WRITE = ("w", "wt", "tw")

    def _open_args(self, call, what):
        """The path and mode of `open(path, mode, encoding=...)`, or a refusal."""
        if not call.args or len(call.args) > 2 or any(isinstance(a, ast.Starred) for a in call.args):
            self.refuse(call, "open with these arguments", "Pass the path and, at most, the mode.")
        mode = "r"
        if len(call.args) == 2:
            m = call.args[1]
            if not (isinstance(m, ast.Constant) and isinstance(m.value, str)):
                self.refuse(m, "a mode that is not a literal", "Write the mode: \"r\" or \"w\".")
            mode = m.value
        for k in call.keywords:
            if k.arg == "mode" and isinstance(k.value, ast.Constant) and isinstance(k.value.value, str) and len(call.args) == 1:
                mode = k.value.value
            elif k.arg == "encoding" and isinstance(k.value, ast.Constant) and isinstance(k.value.value, str) \
                    and k.value.value.lower().replace("-", "").replace("_", "") in ("utf8",):
                pass
            else:
                self.refuse(k.value, "open's `%s=`" % k.arg,
                            "py2viv reads and writes UTF-8 text with Python's newline translation: pass the path, "
                            "the mode and, at most, encoding=\"utf-8\".")
        if mode not in self.OPEN_READ + self.OPEN_WRITE:
            self.refuse(call, "open in the mode %r" % mode, "py2viv reads (\"r\") and writes (\"w\") text files.")
        if what == "read" and mode not in self.OPEN_READ or what == "write" and mode not in self.OPEN_WRITE:
            self.refuse(call, "a file opened in the mode %r and %s" % (mode, "read" if what == "read" else "written"),
                        "Python raises io.UnsupportedOperation.")
        pc, pt, _ = self.expr(call.args[0])
        if pt.kind != "str":
            self.refuse(call.args[0], "a path that is %s" % self._kind_words(pt), "Pass the path as a string.")
        return pc, mode

    def _is_open(self, node):
        return isinstance(node, ast.Call) and self._call_name(node.func) == "open" and self._no_def("open")

    def _is_path(self, node):
        """Path(p) (pathlib.Path, imported by name or through the module)."""
        if not isinstance(node, ast.Call):
            return False
        name = self._call_name(node.func)
        return name in ("Path", "pathlib.Path") and name.split(".")[0] in getattr(self, "pathlib_names", set())

    def _file_method(self, n):
        """open(p).read(), open(p).readlines(), Path(p).read_text(), Path(p).write_text(s),
        open(p, "w").write(s): (code, T, prec), or None when the receiver is neither."""
        f = n.func
        recv = f.value
        m = f.attr
        if self._is_open(recv):
            if n.keywords or (m in ("read", "readlines") and n.args):
                self.refuse(n, "`.%s` with arguments" % m, "Read the whole file: .read() or .readlines().")
            if m == "read":
                pc, _ = self._open_args(recv, "read")
                return self._host_ascii("read_text(%s)" % pc), P.STR, 100
            if m == "readlines":
                pc, _ = self._open_args(recv, "read")
                return "%s(%s)" % (self.use("py_keeplines"), self._host_ascii("read_text(%s)" % pc)), P.STRS, 100
            if m == "write":
                pc, _ = self._open_args(recv, "write")
                if len(n.args) != 1:
                    self.refuse(n, "write with these arguments", "Pass the text.")
                sc, st, _ = self.expr(n.args[0])
                if st.kind != "str":
                    self.refuse(n.args[0], "writing %s" % self._kind_words(st), "Python writes strings (TypeError otherwise).")
                return "write_text(%s, %s)" % (pc, sc), P.NUM_I, 100
            self.refuse(n, "the method `.%s` of a file" % m, "py2viv reads a file whole (read, readlines, a for loop) and writes it with write.")
        if self._is_path(recv):
            if recv.keywords or len(recv.args) != 1:
                self.refuse(recv, "Path with these arguments", "Pass one path.")
            pc, pt, _ = self.expr(recv.args[0])
            if pt.kind != "str":
                self.refuse(recv.args[0], "a path that is %s" % self._kind_words(pt), "Pass the path as a string.")
            for k in n.keywords:
                if not (k.arg == "encoding" and isinstance(k.value, ast.Constant) and isinstance(k.value.value, str)
                        and k.value.value.lower().replace("-", "").replace("_", "") == "utf8"):
                    self.refuse(k.value, "Path.%s's `%s=`" % (m, k.arg), "Pass at most encoding=\"utf-8\".")
            if m == "read_text" and not n.args:
                return self._host_ascii("read_text(%s)" % pc), P.STR, 100
            if m == "write_text" and len(n.args) == 1:
                sc, st, _ = self.expr(n.args[0])
                if st.kind != "str":
                    self.refuse(n.args[0], "writing %s" % self._kind_words(st), "Python writes strings (TypeError otherwise).")
                return "write_text(%s, %s)" % (pc, sc), P.NUM_I, 100
            self.refuse(n, "Path.%s" % m, "py2viv maps Path(p).read_text() and Path(p).write_text(s).")
        return None

    def _file_lines_source(self, it):
        """`for line in open(p)`: the lines with their line ends, as Python iterates a text file."""
        pc, _ = self._open_args(it, "read")
        return "%s(%s)" % (self.use("py_keeplines"), self._host_ascii("read_text(%s)" % pc))

    def _s_With(self, s, top):
        """`with open(p) as f:` / `with open(p, "w") as f:`. Reading: the body may read the file
        once (f.read(), f.readlines(), `for line in f`). Writing: each f.write(s) adds to the text,
        which is written when the block ends (Python's buffered file holds it until then too)."""
        if len(s.items) != 1 or not self._is_open(s.items[0].context_expr) or not isinstance(s.items[0].optional_vars, ast.Name):
            self.refuse(s, "`with`", "The numeric subset has no context managers but `with open(path) as f:`; drop the `with`.")
        call = s.items[0].context_expr
        f = s.items[0].optional_vars.id
        mode = "r"
        if len(call.args) == 2 and isinstance(call.args[1], ast.Constant):
            mode = call.args[1].value
        for k in call.keywords:
            if k.arg == "mode" and isinstance(k.value, ast.Constant):
                mode = k.value.value
        if mode not in self.OPEN_READ + self.OPEN_WRITE:
            self.refuse(call, "open in the mode %r" % (mode,), "py2viv reads (\"r\") and writes (\"w\") text files.")
        writing = mode in self.OPEN_WRITE
        uses = [m for st in s.body for m in ast.walk(st) if isinstance(m, ast.Name) and m.id == f]
        par = self.parents
        reads = 0
        for u in uses:
            p = par.get(u)
            if isinstance(p, ast.Attribute) and p.value is u and isinstance(par.get(p), ast.Call) and par.get(p).func is p:
                if p.attr in ("read", "readlines") and not writing:
                    reads += 1
                    continue
                if p.attr == "write" and writing and isinstance(par.get(par.get(p)), ast.Expr):
                    continue
                self.refuse(p, "the method `.%s` of a file opened in the mode %r" % (p.attr, mode),
                            "py2viv reads a file whole (read, readlines, a for loop) and writes it with write "
                            "(as a statement).")
            if isinstance(p, ast.For) and p.iter is u and not writing:
                reads += 1
                continue
            self.refuse(u, "the file `%s` used as a value" % f, "Read it (f.read()) or write it (f.write(s)) only.")
        if reads > 1:
            self.refuse(s, "a file read more than once in a `with` block",
                        "Python's second read starts at the end (it gives \"\"); read it once into a variable.")
        pc, _ = self._open_args(call, "write" if writing else "read")
        path = self.new_tmp("path")
        self.flush_pre(s)
        self.emit("%s = %s" % (path, pc), s)
        self._open_files = getattr(self, "_open_files", {})
        if writing:
            buf = self.new_tmp("buf")
            self.emit("%s = \"\"" % buf, s)
            self._open_files[f] = ("w", path, buf)
        else:
            self._open_files[f] = ("r", path, None)
        try:
            self._block(s.body)
        finally:
            del self._open_files[f]
        if writing:
            n = self.new_tmp("n")
            self.emit("%s = write_text(%s, %s)" % (n, path, buf), s)

    def _lines_of_file(self, q):
        """The lines a loop over `open(p)` or a `with` file reads (with their ends), or None."""
        if self._is_open(q):
            return self._file_lines_source(q)
        of = self._open_file(q)
        if of is None:
            return None
        if of[0] != "r":
            self.refuse(q, "a loop over a file opened for writing", "Python raises io.UnsupportedOperation.")
        return "%s(%s)" % (self.use("py_keeplines"), self._host_ascii("read_text(%s)" % of[1]))

    def _open_file(self, node):
        """The `with` file a name holds: (mode, path code, buffer code) or None."""
        if isinstance(node, ast.Name):
            return (getattr(self, "_open_files", None) or {}).get(node.id)
        return None

    def _with_file_method(self, n):
        """f.read(), f.readlines() and f.write(s) of a `with` file."""
        f = n.func
        of = self._open_file(f.value)
        if of is None:
            return None
        mode, path, buf = of
        if f.attr == "read" and not n.args and not n.keywords:
            return self._host_ascii("read_text(%s)" % path), P.STR, 100
        if f.attr == "readlines" and not n.args and not n.keywords:
            return "%s(%s)" % (self.use("py_keeplines"), self._host_ascii("read_text(%s)" % path)), P.STRS, 100
        self.refuse(n, "`.%s` of a file with these arguments" % f.attr, "Read the whole file: .read() or .readlines().")

    def _with_write_stmt(self, call, s):
        """f.write(s) as a statement in `with open(p, "w") as f`: the text is added to the buffer."""
        f = call.func
        if not (isinstance(f, ast.Attribute) and f.attr == "write"):
            return False
        of = self._open_file(f.value)
        if of is None or of[0] != "w":
            return False
        if len(call.args) != 1 or call.keywords:
            self.refuse(call, "write with these arguments", "Pass the text.")
        sc, st, sp = self.expr(call.args[0])
        if st.kind != "str":
            self.refuse(call.args[0], "writing %s" % self._kind_words(st), "Python writes strings (TypeError otherwise).")
        self.flush_pre(s)
        self.emit("%s = %s + %s" % (of[2], of[2], self._paren(sc, sp, 61)), s)
        return True

    @staticmethod
    def _str_only_printed(body, name):
        """A loop over literal rows (py2viv.py's _unroll) may write a row's string constant where
        the body reads the loop name, when the body never rebinds the name."""
        return not any(isinstance(m, ast.Name) and m.id == name and not isinstance(m.ctx, ast.Load)
                       for st in body for m in ast.walk(st))

    # ------------------------------------------------ printing
    def _str_print_forms(self, node, code, t, part):
        spec = part.get("spec")
        render = part.get("render")
        if t.kind == "strs":
            if spec:
                self.refuse(node, "a format spec on a list", "Python raises TypeError.")
            return [code if not code.startswith('"') else "(%s)" % code]
        if render == "repr":
            code = "repr(%s)" % code
        if spec is not None and spec != "":
            d = parse_spec(spec)
            if d is None:
                self.refuse(node, "the format spec `%s`" % spec, "Python raises ValueError (an invalid format specifier).")
            code = self._str_spec(node, code, d, spec)
        if code.startswith('"'):
            code = "(%s)" % code  # a value, never merged with the text beside it
        return [code]
