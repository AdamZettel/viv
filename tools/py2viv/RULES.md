# py2viv translation rules

py2viv's mappings from NumPy, the math module and Python builtins to Viv are data, in
`rules/*.toml`. The translator (`py2viv.py`) still parses Python with `ast`, infers static
types, translates statements and operators, and emits Viv. When it reaches a call of a
function, a method or attribute of an array, a module constant, or an unpacking such as
`q, r = divmod(a, b)`, it asks the rule engine (`rule_engine.py`). The engine finds the rule,
binds the arguments and runs the rule's steps.

The format is narrow on purpose. It is not a program-transformation language: a rule rewrites
one call, attribute or unpacking whose arguments the translator has already parsed and typed.
Parsing stays per language (Python's `ast` here; tree-sitter for a later front end), and so
does everything that is not a library mapping.

```
rules/strings.toml    Python's str: str, repr, format, chr, ord, and len, float, int, list of a
                      string; the methods of str and of a list of strings (strs.py has the rest)
rules/builtins.toml   Python builtins and the math module
rules/numpy.toml      NumPy, array methods and attributes, module constants, unpackings
rules/helpers.toml    the Viv defs a translation may call: the prelude, the guards that stop
                      where Python raises, NumPy's and CPython's orders of summation, and the
                      library modules they carry (lib/nprandom, a helper's `file`)
rules/modules.toml    the hint for each refused import (scipy, torch, random, sys)
rule_engine.py        loader, matcher, binder, the expression language, the hooks
strs.py               strings: literals, the text of values (str, repr, f-strings, str.format,
                      %-formatting), operators, subscripts, join, lists of strings, files
views.py              where a NumPy view (a slice, a row, a transpose, a reshape) could be
                      observed, and which values NumPy holds in a transpose's memory order
probe_numpy_order.py  NumPy's order of summation, checked bit for bit (results/axis_order.txt)
check_rules.py        every rule's generated differential test, the static checks, the ledger
check_rules_equiv.py  the proof that this translates as the table-in-code translator did,
                      except where a listed hole was fixed
```

`python3 tools/py2viv/rule_engine.py` checks the files and prints the counts;
`--table` prints the rule tables that SUBSET.md carries (generated; `check_rules.py` fails
when SUBSET.md differs from them).

## A rule

```toml
[[rule]]
id = "np.linalg.solve"
match = "np.linalg.solve(a, b)"
exactness = "tolerance"
rtol = 1e-12
atol = 1e-13
evidence = "adapters.py:37 sol: bitwise (3x3); held to adapters.py's declared rtol 1e-12, atol 1e-13"
note = "Viv's LU is its own, not LAPACK's dgesv."
cases = [
  { inexact = "{name} (Viv's own LU, Cholesky, Jacobi or FFT, not LAPACK or pocketfft)" },
  { eval = ["a", "b"] },
  { if = "not (a.kind == 'arr' and a.rank == 2 and b.kind == 'arr' and b.rank in [1, 2])", refuse = "np.linalg.solve of {a.words} and {b.words}", hint = "Pass a real square matrix and a real vector or matrix." },
  { if = "b.rank == 1", use = ["py_solve"], viv = "py_solve({a}, {b})", type = "arr(1)" },
  { use = ["py_solve_multi"], viv = "py_solve_multi({a}, {b})", type = "arr(2)" },
]
tests = [{ a = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]], b = [1.0, 2.0, 3.0] }]
```

| field | meaning |
|---|---|
| `id` | a name for reports; unique |
| `match` | one pattern or a list (below) |
| `context` | `expr` (default), or `stmt` for a method call that stands as a statement (`a.sort()`) |
| `params` | how a parameter binds when it is not an expression: `shape` or `dtype` |
| `when` | a condition on the call's syntax; false means the rule declines and the next is tried |
| `cases` | the steps, in order (below) |
| `exactness` | `exact`, `tolerance` or `refuse` (below); required |
| `rtol`, `atol` | a tolerance rule's bound, which a program using the rule must declare to diff.py |
| `evidence` | where the class comes from: SUBSET.md's "What is exact", a run in `results/`, or the reason |
| `gap`, `lift` | a refuse rule: the language gap (an R-number from the idioms study) or the change that would lift it |
| `guard` | the values for which the class holds, where they are narrower than all inputs and nothing checks them yet |
| `tests` | the seed inputs of the rule's generated differential test (`check_rules.py`): one table per test, one key per parameter; numbers, Bools, nested arrays, the strings `"inf"`, `"-inf"`, `"nan"`, a complex number as a string (`"1.5-2j"`), a dtype as its name (`"int"`), `[]` an empty Python list (its variants are a NumPy array and a non-empty list), a Python string as `{ str = "..." }` (its variants are every kind of text, a number and a list); `only = ["np.dot"]` limits a test to some spellings. Every rule that translates has at least one |
| `note` | a known limitation, kept with the rule |
| `vars` | constants the rule's expressions can read (used by families) |
| `setup` | Python lines check_rules.py puts before each of the rule's generated cases (`np.random.seed(12345)`, `@g = np.random.default_rng(12345)`); a test may name what the setup binds (`g = "@g"`) |

## Patterns and binding

A pattern is the Python form, with a Python parameter list:

| pattern | kind | matches |
|---|---|---|
| `np.linalg.solve(a, b)` | call | the dotted name, however numpy was imported (`np`, `numpy`, `from numpy.linalg import solve`) |
| `np.roll(a, shift, axis=None)` | call | `axis` may be omitted, passed third or by keyword |
| `np.sqrt(x, /, *out)` | call | `x` positional only (a keyword `x=` is refused, as NumPy refuses it); `out` collects further positional arguments, for the rule to refuse |
| `math.isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0)` | call | `rel_tol` and `abs_tol` only by keyword |
| `f(x, **kw)` | call | keywords not in the list are readable as `kw_<name>` (no rule uses one now) |
| `np.linalg.det(...)` | call | any arguments, none bound (refusals, which translate nothing) |
| `np.random*(...)`, `np.emath.*(...)` | call | `*` matches any characters, dots included; tried after every rule spelled out for the name has declined. Only a `refuse` rule may have one: a translating rule names each function it maps, so a name with another module path in the middle (np.emath.sqrt, np.linalg.sqrt) cannot reach it |
| `a.dot(b)` | method | a first name that is not `np`, `math` or `random` is the receiver |
| `a.T` | attr | an attribute of an array (or number) |
| `np.pi` | const | a module attribute |
| `q, r = divmod(a, b)` | unpack | a tuple assignment of the call |
| `a[lo:hi:step]` | slice | one range of a subscript (`a[1:3]`, the `1:-1` of `A[1:-1, j]`); py2viv.py splits a subscript into its axes and gives each range to the rule, with `n` (the Viv code of that axis's length) and `axis`; `lo`, `hi` and `step` are each passed or not (`a[None:3]` is `a[:3]`) |
| `x + y`, `x < y` | binop | an operator of two arrays (`+ - * /` and the six comparisons), both operands translated; `op` is the operator and `rt` the result's type the operator's typing found (its `.kind`, `.rank`, `.py`, `.container` can be read) |
| `a[i]`, `a[:, i]`, `a[i, j]` | index | a subscript with an array among its items (a Bool mask or an integer array); the pattern's items are `:` (a whole range) or names, and py2viv.py matches the subscript's shape with trailing whole ranges left out (`A[m, :]` is `a[i]`); every named item is bound, `axis` is the position of the first and `nitems` the number of items |
| `a[i] = v` | index-store | a store through such a subscript, `v` the value; `a[i] += v` is the read, the operation and this store |
| `a[None]` | newaxis | a subscript with `None` or np.newaxis among its items: `a` is the subscript without them, `axes` the positions of the new axes in the result (Viv text, `[0, 2]`) and `nnew` their number |
| `np.nonzero(a)[k]` | item | one item of a call's tuple result: the call's arguments bind as a call's, `k` binds the subscript |

Rules are tried in file order (strings.toml, builtins.toml, then numpy.toml), exact names before
`*` patterns. The first rule that does not decline translates the call or refuses it. A rule that
declines (`next`) after translating arguments to read their types hands the translations to the
next rule, so each argument is translated once (its hoisted statements and inexact operations
are not doubled). A string argument that reaches a rule outside strings.toml whose steps do not
name `'str'` is refused there (Python raises TypeError, or, for min and max, compares strings). A call no
rule takes is refused with the messages the translator always gave: "`np.foo`: Not in the
NumPy mapping yet" for a NumPy or math name, "a call of `foo`" otherwise; a method no rule
takes, "the method `.m`"; an attribute, "the attribute `.m`".

**Binding** is Python's, strict: positional arguments bind the parameters in order, a keyword
binds the parameter it names (a positional-only one, before `/`, takes none), and the call is
refused for a starred argument, `**kwargs`, a parameter passed twice, and a required
parameter missing ("np.linspace without its argument `start`", where Python raises
TypeError). A positional argument past the parameters (with no `*rest`) and a keyword the
signature does not name (with no `**kw`) are kept aside: the rule's steps can refuse them with
their own message (`nargs`, `extra_kws`), and if none does, the rule refuses them before it
translates ("np.trace with a second positional argument"). No argument is ever dropped.
Patterns carry the Python function's own parameter names where it accepts keywords
(`np.zeros(shape, dtype=None)`: `np.zeros(3, int)` binds the dtype), and `/` where it does not
(builtins, math, NumPy's ufuncs), so a keyword Python would reject is refused. A parameter
that is not passed reads as its default (`k.kind` of an omitted `k=0` is an int), or, for a
default of None, as the type `none` with code that is not Viv (so a template that writes it
without testing `.given` fails to build).

A parameter with `params = { p = "seq" }` is a tuple or list literal of arrays
(`np.concatenate((a, b))`): each entry is translated, `{p}` is their codes joined by commas, and
the functions `seq_n`, `seq_kinds`, `seq_ranks`, `seq_py` and `seq_codes(p, 'helper')` read
them; a sequence held in a variable is refused.

A parameter with `params = { a = "list" }` binds as an expression; check_rules.py writes its test
values as Python lists (the receiver of a list method), where it writes NumPy arrays otherwise.
A parameter with `params = { spec = "literal" }` binds as an expression too, and check_rules.py
writes its test values in the call, as literals, where it binds every other value to a name
first: the rule reads it from the syntax (a format spec, `indexing='ij'`).

A parameter with `params = { p = "shape" }` is translated as a shape (a number or a tuple of
numbers: `{p}` is the dimensions, joined; `p.dims` the list, `p.ndims` its length). One with
`"dtype"` is not translated: its value is the dotted name (`float`, `np.float64`) or the
constant written.

## Steps

`cases` is a list run in order. An entry applies when its `if` holds (or it has none). Each
entry does one or more of:

| field | does |
|---|---|
| `eval = ["x", "y"]` | translates these arguments now, in this order |
| `keywords = ["dtype"]`, `refuse`, `hint` | refuses the first keyword not in the list, at its value; `{kw}` is its name |
| `inexact = "..."` | records an operation whose Viv result can differ from Python's in the last bits; the manifest lists it and diff.py's report shows it |
| `refuse`, `hint`, `gap`, `at` | refuses: the message and hint are templates; `at` names the argument the caret goes under (default the whole call; `targets` for an unpacking's left side) |
| `use = [...]` | the helpers the translation calls (from helpers.toml) |
| `tmps = ["x"]`, `hoist = ["{tmp_x} = {x}"]` | fresh temporaries (`{tmp_x}` is `py_x<n>`), and statements placed before the current one |
| `viv`, `type`, `prec` | the result: a Viv template, its static type, its precedence (default 100) |
| `python = "x ** y"` | the result: the call rewritten as this Python expression, whose names are the bound arguments; the translator translates it (an unpacking rule: `python = "(a // b, a % b)"`, and the translator translates the assignment) |
| `hook`, `args` | the result: a named hook (below) |
| `const = "pi"` | the result: a numeric constant, written as the translator writes a float literal |
| `next = true` | declines: the next rule is tried (a statement rule gives the call back to expression translation) |

An entry without a result (`viv`, `python`, `hook`, `const`) is a step: it evaluates,
records, uses or refuses, and the next entry runs. An entry with a result ends the rule. An
entry may override `exactness`, `rtol`, `atol` and `evidence` for the value it produces
(abs of a complex number is a tolerance; abs of a real is exact).

**When arguments are translated.** Translating an argument can emit statements before the
current one (a conditional expression, a generator) or be refused, and the manifest keeps the
order in which inexact operations are recorded, so the order is part of the output. If a rule
has no `eval` step, all its parameters are translated in pattern order before the first
entry. If it has one, nothing is translated before it, and an argument not yet translated is
translated when a condition or template first reads its code or type (its syntax, `x.syntax`,
`x.lit`, `x.strlit`, `x.callname`, `x.given`, is read without translating it). Each argument
is translated at most once. A required argument that is missing is refused, located at the
call.

**An argument written twice.** A template that writes an argument's code more than once
(`{x:71} * {x:71}`, `{x:51} != {x:51}`, a `python` rewrite naming it twice) evaluates it
more than once. That is the same only if evaluating it has no effect, so the rule is refused
when the argument's Viv code calls a def of the program (which may print or write into an
array): "np.square of an expression that calls `f`", with the hint to assign it first.

## Expressions and templates

Conditions (`if`, `when`), `type` and `prec` are Python expressions over a whitelist: names,
constants, `and`, `or`, `not`, comparisons (`in` included), `a if c else b`, `+`, `-`, list
literals, subscripts and slices, and calls of `arr`, `T`, `num_of`, `dims_like`, `paren`,
`len`, `all_int`, `size`, `sizes`, `seed_words` (SeedSequence's 32-bit words of a seed written with int
constants, as a Viv array literal, or None), `max`, `pure` (no argument's code calls a def of the program, so writing it twice, or reading it again, changes nothing), `vec` (the argument's code, or a number as a vector of one entry, which broadcasts to any shape), `same_shape` (two arguments whose values have one shape by their form, py2viv.py's `_shape_key`: one array, ranges of it with literal bounds that trim as much, `np.roll` or an elementwise function of it, arithmetic of these with numbers), `axk` (the axis a literal `axis=` names, 0 to rank - 1,
or -1 when none is), `kd` (1 for `keepdims=True`), the `seq_*` functions above (a size, or a shape's sizes, through `py_size` unless it is a
literal or a `len`/`numel`/`shape`, since NumPy raises for a negative size and Viv makes an
empty array) and `py_of` (NumPy's result dtype of real operands: int, bool or float). The names
are:

| name | value |
|---|---|
| `op`, `rt` | a binop rule's operator and result type |
| `axis`, `nitems` | an index rule's first array position and number of items |
| `axes`, `nnew` | the newaxis rule's positions (Viv text) and count |
| an argument `x` | `x.kind` (`num`, `bool`, `arr`, `carr`, `cnum`, `str`, `unknown`, ...), `x.rank`, `x.py` (`int`, `float`, `bool`, `mixed`), `x.container` (`nd`, `list`, `shape`), `x.arr`, `x.scalar`, `x.words` ("a rank-2 array"), `x.code`, `x.prec`; syntax: `x.given`, `x.syntax` (the `ast` class name), `x.lit`, `x.strlit` (a string literal, stripped and lower-cased), `x.callname`, `x.grows` (a name bound only to `[]` that nothing can alias: `grow_refusal` in py2viv.py), `x.axis_lit` (an int literal, a negative one too, else None), `x.transposed` (`X.T`, `np.transpose(X)` or `X.transpose()` of a matrix X in row order), `x.f_order` (the value may be held in a transpose's memory order: views.Order); a shape: `x.dims`, `x.ndims`, `x.dims_int` |
| `name`, `short`, `module` | the matched name (`np.linalg.solve`), its last part, its first |
| `nargs`, `nkw`, `kw_names`, `extra_kws` | the call's counts and keyword names |
| `in_grad` | the call is inside a def that `grad` or `value_and_grad` differentiates |
| `np_seeded` | the module calls `np.random.seed` somewhere (np.random's legacy draws need it) |
| `host_py` | the running CPython's version, `[3, 12]` (a list sum differs from 3.12 on) |
| `ntargets`, `targets_named`, `targets_plain` | an unpacking's left side |
| `NUM_I`, `NUM_F`, `BOOL`, `CNUM`, `UNK` | the translator's scalar types |
| `arr(rank, py="float", container="nd")`, `T(kind, rank, py, container)` | array types |
| a `dtype` parameter, a `vars` entry, `tmp_<stem>` | its value |

A type expression that is an argument (`type = "x"`) means that argument's type.

In a template (`viv`, `refuse`, `hint`, `inexact`, `hoist`, `use`, hook `args`), `{x}` is
argument x's Viv code, `{x:71}` the code parenthesized unless its precedence is at least 71
(70 is `*`, 60 is `+`, 50 a comparison), `{x|0}` the code or `0` when x was not passed, `{{`
and `}}` braces, and any other `{...}` an expression whose value is written with `str`.

**Precedence** (checked by `check_rules.py`, the precedence audit): a template that is one
argument and nothing else passes that argument's precedence on (`viv = "{x}", prec =
"x.prec"`), so `np.transpose(v + 1.0) * 2.0` keeps its parentheses; an argument written next
to an operator outside parentheses is written `{x:N}` with N at least the case's own `prec`
(the right operand of `^` at least 80); a template with such an operator declares its `prec`.

## Adapters that need several Viv statements

Three ways, none with code in the rule:

- **A helper def.** The template calls a def from `rules/helpers.toml` and names it in `use`;
  the translation carries the def (and its `deps`) once, in the file's order. The
  calling-convention adapters work this way: `np.sort` is `py_sort(a)`, which copies, checks,
  sorts the copy in place and returns it; `np.linalg.solve`, `cholesky`, `inv`, `eigvalsh`
  and `np.fft.fft`/`ifft` likewise. So do the guarded math functions (`math.sqrt` is
  `py_msqrt`, which stops where CPython raises ValueError).
- **A library module.** A helper with `file = "lib/nprandom/pcg64.viv"` instead of `viv` is
  that module's text without its `###` header, its `#!` lines and its `test` blocks; its
  `deps` stand for its `#! use` lines. The translation stays one self-contained file (it runs
  from a temporary directory, where `#! use` would find nothing), and the library stays the one
  source: np.random's rules call lib/nprandom this way.
  A helper whose `stop` states only the start of NumPy's message (NumPy's rest is computed:
  the shapes, the index) lists that message in `stops`; the manifest carries it, and diff.py
  compares such a stop with the start of Python's last line (`ValueError: operands could not
  be broadcast together with shapes` against `... with shapes (3,) (4,) `).
- **Hoisted statements.** `tmps` and `hoist` put statements before the current one:
  `math.sinh(x)` evaluates x once into a temporary and checks the result for overflow.
- **A rewrite into Python.** `python` hands a Python expression back to the translator:
  `q, r = divmod(a, b)` is `q, r = a // b, a % b`, and `pow(a, b)` is `a ** b`, each then
  translated by the operator rules in py2viv.py (which are exact for ints, guarded for zero
  divisors, and so on).

## Hooks: the mappings still in code

A case can name a hook, a Python function in `rule_engine.py`. There are four:

| hook | used by | why it is code |
|---|---|---|
| `reduce_gen` | `sum`, `all`, `any` of a generator expression | it translates the generator's `for` and `if` clauses into an accumulator loop hoisted before the statement: statement translation, which a template cannot do |
| `fold_minmax` | `min(a, b, c, ...)`, `max(...)` | any number of arguments, folded pairwise, with a refusal at the first array or at an int beside a float: a template has a fixed number of arguments |
| `np_maxmin` | `np.maximum`, `np.minimum` | the helper depends on which operands are arrays (`py_npmaxv`, `py_npmaxs`, `py_npmax2`), and two ints need none |
| `grow_append` | `lst.append(x)` of a list py2viv grows (`list.append-statement`) | the list's element type, int or float (Python prints them differently), or its rows' shape, is the translator's state: the first append sets it, every later one must match, and the `[]` and every read of the list are typed by it; a row appended is checked to be the list's own (funcs.py) |
| `list_method` | `lst.pop(i)`, `insert`, `extend`, `remove`, `clear`, `reverse`, `sort(key=, reverse=)`, `index`, `count` (the rules under "Python lists" in numpy.toml) | as `grow_append`: the element type and shape are the translator's state, a method that changes the list rebinds its name (funcs.py, `ListMethods._list_method`), and `sort` sorts a permutation by keys that a lambda or a def computes |
| `str_text` | `str(x)`, `repr(x)` (strings.toml) | the text follows the static type in more ways than cases say well: an int's digits, a float's repr, True/False, a string or its repr, a list as Python prints it (strs.py, `_str_of`) |
| `str_format` | `format(x, spec)` | the spec is read as Python reads it and split into Viv's number format and a padding (strs.py, `_str_format`) |
| `str_join` | `sep.join(x)` | a generator or a comprehension is translated as a list first, a string as its characters (strs.py, `_str_join`) |
| `strs_method`, `strs_find` | `lst.pop`, `insert`, `extend`, `remove`, `clear`, `reverse`, `sort`, `index`, `count` of a list of strings | a Viv list of strings is never changed in place: each method rebinds the list's name, under the growable-list proof (`a.grows`) |

Mappings that stay in `py2viv.py`, because they are statement, operator or name translation,
not a library call:

- the operators (`/`, `//`, `%`, `**`, `@`, bit operations) and their guards; an operator of two
  arrays (`+ - * /`, the comparisons) is handed to the binop rules after the operator's typing
  and refusals (Bools, lists, shapes, complex numbers) ran;
- ints annotated `int` as Viv Ints, enum.Enum classes as Viv enums and `Optional[...]` as `T?`
  (`ints.py`, SUBSET.md "Ints, enums and optionals"): the operators of Ints and their guards
  (`py_ifloordiv`, `py_imod`, `py_idiv` in helpers.toml), `str`, `abs`, `min`, `max`, `float`,
  `int` and `round` of Ints, a member's text, and the `match` an `is None` test becomes. An Int
  passed to a rule is first converted to a number exactly (`n + 0.0`, which stops past 2^53),
  so every rule sees the Python int kept as a double it always saw, and no rule changed;
- `assert abs(a - b) < tol`, `assert math.isclose(...)` and `assert np.allclose(...)`, which
  become `assert_near`/`assert_true` (the expression forms of isclose and allclose are rules);
- the receiver checks of a method call (a string, a non-array, keyword arguments), before the
  method rules;
- `lst = []`, and `lst = [a, b]` of a list the scope appends to (R17): the whole scope is
  checked for any use that could alias the list (`grow_refusal`), and the binding is refused,
  with that use named, unless every binding is a new list (`[]`, a list literal of all ints or
  all floats, a comprehension, `list(...)`, `sorted(...)`, `[x] * n`, `.copy()`), the list is
  bound in this scope (not a parameter, not another scope's list), and every other use is
  `.append(x)`, another list method (funcs.py: `LIST_MUTATORS` as a statement, `pop` also as
  the whole value of an assignment or a return, `index`, `count`, `copy` anywhere), `len`,
  indexing, a `for` or a comprehension over it (or its `enumerate`, `reversed`, `zip`) that does
  not change it, `return`, a condition, arithmetic or a comparison, or an argument of `len`,
  `sum`, `min`, `max`, `sorted`, `print`, `list` or a NumPy function. The appends are the rule
  `list.append-statement`, the other methods the rules under "Python lists", which read this
  proof as `a.grows`;
- funcs.py: the rewrite of nested defs, `f = lambda`, lambdas passed to defs, `map` and
  `filter` before translation, and the translation of comprehensions, generator expressions
  (and the builtins that read them), calls of lambdas and of function parameters, key
  functions, lists of rows and of tuples, `in` and `==` of lists, a negative step of a vector's
  slice, and text built in a print (SUBSET.md, "Functions and lists"): statement translation,
  which a rule's template cannot do;
- `from numpy import pi` (a name bound to a constant), and which bare names are builtins;
- `range`, `enumerate`, `zip` and `reversed` in a `for` header, `print` and `fmt`, `repr` and
  `.tolist()` in a print, `value_and_grad`/`grad`, `n, m = A.shape`;
- which subscript is which: one with `None` goes to the newaxis rule, one with an array among
  its items (found by translating the items once, the receiver first) to the index rules, with
  its shape as the key (`_fancy`: a range beside the array, `None` or `...` beside it, a mask
  beside another index are refused there);
- a subscript with a range (`a[1:3]`, `A[:, j]`, `A[i][1:]`): each range through the rule
  `slice`, each plain index as NumPy reads it (a negative literal from the end, any other
  through `py_ix`), a subscript of a slice (`a[1:3][0]`) refused; a store into a slice or a row
  (`a[1:3] = v`, `A[i] = v`, `a[1:-1] += v`), whose array value NumPy broadcasts into the slice's
  shape (`py_bset`, which stops where it cannot; a value of a higher rank is refused);
- the view check (views.py): a slice, a row, a transpose, the real or imaginary part of a
  complex array, `np.reshape` and `ravel`, and a def that may return one of them, share memory
  with their array in NumPy and are copies in Viv. Bound to a name (or looped over), one is
  refused when anything in the scope, before or after, writes into the array or into the view:
  a store, `+=` of the whole array, an in-place method, a second name for either written, a def
  called with either that writes into that argument (found for every def, to a fixed point),
  a def that writes the module's array. Used in place (an operand, an argument), one is
  refused when its statement also calls a def that writes into an array, or when it is the
  receiver of an in-place method. A range of a rank-1 Python list is a copy in Python too, and
  a subscript with a mask or an integer array is a copy in NumPy (views.is_fancy); `None` in a
  subscript makes a view;
- `x += v` of an array another name, a caller or a view may hold (views.aliased): NumPy
  changes the array in place, so the translation writes `x[:] = py_fit(x + v, x)` where Viv's
  `x += v` would rebind x; with an array v, `py_inpl(py_badd(x, v), x)`, which stops where the
  broadcast result is larger than x (NumPy raises); a result x's dtype cannot hold is refused;
- `_param_type`'s guess of a parameter's type from the functions its body calls (a parameter
  whose attributes only one class has is an object of that class);
- classes, dataclasses, named tuples and dicts with literal keys (`classes.py`, a mixin of the
  translator): Viv records, their constructors and methods, printing, and the state threading
  of fields rebound after construction. They are not library calls: they need the translator's
  state (the record declarations and their field types across passes, which defs change an
  object passed to them, where an object could get a second name), so they are code, not rules.
  `check_rules.py` still generates their differential tests, as pseudo-rules `objects.*`
  (SUBSET.md, "Classes, dataclasses, named tuples and dicts");
- dicts whose keys are computed at run time (`maps.py`, a mixin of the translator before
  classes.py): Viv maps. Which dict becomes a map is a scan of each scope (an empty dict,
  `dict()`, a comprehension, a defaultdict or Counter, a literal with a computed or float key,
  or a literal of int keys whose variable is used as a map); its key and value types are the
  translator's state across passes; a loop over it needs the body's changes to a dict (found
  through the defs it calls) to pick the checked loop or refuse; so they are code, not rules.
  `check_rules.py` generates their differential tests as pseudo-rules `maps.*` (SUBSET.md,
  "Dicts with keys computed at run time");
- strings (`strs.py`, a mixin of the translator): literals and the Unicode check, the text of values
  (`str`, `repr`, f-strings, `str.format`, `%`-formatting, format specs), `+` `*` the comparisons
  `in` and truth of strings, subscripts and slices (Python's positions are Viv's for text), `join`
  of generators, lists of strings (their appends are `list.append-statement`, the rest the rules
  `strs.*`), loops over strings, files read whole and written in a `with`, a computed exception
  message, and printing strings: expression and statement translation, and state (a `with` file,
  the list's element type), so code, not rules. `check_rules.py` generates their differential tests
  as pseudo-rules `strings.*` (SUBSET.md, "Strings");
- a Generator (`np.random.default_rng`), the static kind `rng`: its Viv value is lib/nprandom's
  state array, so it may be bound to a name, passed to a def, returned and drawn from, where
  Python's object semantics and Viv's shared array agree; `_rng_use` refuses any other use, a
  method no rule takes and an attribute. `np_seeded` (a scan of the module for
  `np.random.seed`) and the calls that draw (`_draws`, in the hoist-order check).

## NumPy's order of summation

A float sum (and a mean, variance, standard deviation) is exact only in NumPy's order, which
`probe_numpy_order.py` states as Python code and checks against NumPy bit for bit
(results/axis_order.txt): from 0.0, a contiguous run in blocks of 8192 entries (NumPy's
buffer), each by the pairwise sum (py_blk, py_pw); along the last axis that per output entry;
along any other axis the slabs added in order (py_sum_fl). A slice with positive steps sums
as its copy does, flat or along an axis. A transpose, and anything computed elementwise from
one, is summed in memory order, which its row-order copy is not: `np.sum(A.T)` is translated as
the sum of A (with the axes swapped for `axis=`), and a sum, mean, variance or product of
anything else that may hold a transpose's order is refused (views.Order follows names,
parameters through every call, and what defs return).

## Exactness

Every rule states one class, taken from SUBSET.md's "What is exact" and diff.py's runs:

- **`exact`**: the Viv computes the same IEEE operations in the same order as the reference
  (NumPy's pairwise sum, CPython 3.12's compensated sum, Python's order for max and min, the
  signs of zero), and diff.py showed bitwise agreement on every value it printed (`evidence`
  names the run). Where Python raises, the translation stops with the same line (a guard in
  helpers.toml), which is part of being exact.
- **`tolerance`**: the result can differ in the last bits: Viv's libm against the
  platform's, BLAS's order of summation, LAPACK against Viv's own LU, Cholesky, Jacobi and
  FFT, `log(x) / log(b)` for `log10`, `sqrt(x*x + y*y)` for `hypot`. Every such rule records
  an `inexact` operation, so the manifest and diff.py's report list it. `rtol` and `atol` are
  the bound diff.py measured, or, where every measured value agreed bit for bit, the tolerance
  the example program declared and passed. A program that uses the rule declares at least
  this tolerance (`# py2viv: rtol=... atol=...`); diff.py compares bit for bit otherwise.
- **`refuse`**: the rule only refuses. Its message and hint are the refusal; `gap` or `lift`
  says what would lift it. (A refuse rule under `np.linalg` records the inexact operation
  before it refuses, as the old translator did.)

`guard` states values for which the class holds where a helper checks them at run time
(`py_sort` stops on a NaN, and on 0.0 with -0.0). The earlier guards are gone: `hypot` scales
where the squares would overflow (`py_hypot`), `np.sign` of an array gives NaN for NaN
(`py_signs`).

## Strict by construction: check_rules.py

`python3 tools/py2viv/check_rules.py` fails the build (www `npm run check`, posts/check.sh)
unless every rule proves itself:

- the rule files load; every translating rule has `tests`; every parameter a pattern binds is
  read by the rule; the precedence audit passes; SUBSET.md's rule tables are the generated
  ones;
- every rule's generated differential test passes. From each rule's `tests`, for every
  spelling of every pattern, it writes the seed cases, each argument as an expression inside
  an expression (`-f(x - 0.5)`, `f(x - 0.5) * 3`), each argument at 0, -0.0, its negative,
  NaN, inf, -inf, as an int, a Bool, a complex number, a Python list and at another rank, each
  size negative, zero and a float, each dtype in turn, each keyword by keyword, and one
  argument too many, one unknown keyword and one missing; an `axis=` at every axis of the
  array at ranks 1 to 3, negative ones, one past each end and None, with and without
  `keepdims=True`; a sequence (`seq`) as ints, floats and complex arrays, one, three and no
  arrays, beside an array of another rank, a number, an empty vector, a Python list, and held
  in a variable. A slice rule's cases put its range on each axis of ranks 1 to 3 beside ranges
  and plain indices (negative, a variable, out of bounds, a float, `None`, `...`), vary each
  bound (omitted, 0, inside, at and past either end, negative, reversed, None, a variable, a
  float, a Bool, an array) and the step (1, 2, 3, 0, -1, a variable, a float), and add the
  view cases: the slice bound to a name, then written through, written into its array, handed
  to a def that writes, looped over while its array is written, returned by a def (each must
  be refused, or print what Python prints), and only read (each must translate and pass: a
  case with `expect = "pass"` fails if it is refused). CPython with NumPy runs each case;
  the translation runs in the three engines (diff.py), in batches, cached in `.rule_cache/`.
  A case passes when the values agree bit for bit (an `exact` rule, and a case whose
  translation records no inexact operation) or within the rule's `rtol` and `atol`, with the
  same type (int, float, Bool, complex, rank, list or array); when Python raises and Viv
  stops with the same line (or Viv's own error, in every engine); when Viv stops with a
  `py2viv:` message (a run-time refusal); or when the translator refuses the case at the call.
  A rule's seed tests may not be refused unless Python raises. A `refuse` rule's cases must
  be refused with the rule's message. Anything else fails: a crash, a refusal away from the
  call, another value or type, engines that disagree, a run that ends otherwise.
- The string operations translated in code (strs.py) are pseudo-rules too: `strings.operators`,
  `strings.subscripts`, `strings.text`, `strings.lists`, `strings.print`, `strings.files`, whose cases
  put every kind of text (empty, beyond ASCII, non-BMP, combining, every space and line break,
  control characters, quotes) and the edge numbers through each construct, and require CPython's
  printed text byte for byte.
- The typing of defs (infer.py) is no library call, so it has no rule; four pseudo-rules stand
  for it (`infer.clone`, `infer.list`, `infer.columns`, `infer.hints`), as `objects.*` stand for
  classes.py. Their cases call one def at two places with a value of each pair of kinds (ints,
  floats, -0.0, NaN, the infinities, 2**52, Bools, int and float arrays, a matrix, lists of
  floats and of ints) for bodies that return, double, add, negate and print it; change a list
  parameter (append, pop, insert, extend) called with each form of new list and with lists the
  caller still holds; read, loop over, store into and print rows mixing ints and floats; and
  call annotated parameters with each kind of value. They demand one thing more than a rule's
  cases: the printed text must be CPython's byte for byte (an int printed as `3.0` fails, where
  a rule's case compares the number).
- The last run is the ledger, `results/rule_tests.json` (and `results/rule_tests.txt`), with a
  hash of the rule files, the translator and the generator. Where NumPy, viv, cc, node and
  viv.wasm are all present the tests run; elsewhere (CI's site job) the ledger must be current
  and without a failure.

## Families

A `[[family]]` is a list of rules that share their `cases` (and `when`, `params`): the
one-argument elementwise functions, where only the Viv builtin (`vars.fn`), the guarded form
of the math function (`vars.mpath`, `vars.mhelper`), the patterns, the exactness and the tests
differ. Each `[[family.rule]]` is an ordinary rule with the family's fields filled in.

## Fixed holes

The rules first reproduced the table-in-code translator exactly, including what it got wrong.
These holes were fixed on the dates in the table; `check_rules_equiv.sh` proves the translation is otherwise
unchanged (each changed translation is traced to a fix below, by id). The first ten were found
while moving the mappings into rules; the rest by `check_rules.py`'s generated tests.

| id | fixed | found by | the hole | now |
|---|---|---|---|---|
| precedence | 2026-09-27 | moving the mappings into rules | a template that is its argument alone kept precedence 100: `np.transpose(v + 1.0) * 2.0` was `v + 1.0 * 2.0`, `np.real(x + 2.0) * 3.0` was `x + 2.0 * 3.0`; the same in `v.transpose()` of a vector, `round` and `math.floor`/`ceil` of an int | the template passes the argument's precedence on (`prec = "x.prec"`); the precedence audit in `check_rules.py` checks every template |
| log-out | 2026-09-27 | moving the mappings into rules | `np.log(x, b)` was a logarithm to base b; NumPy's second argument is `out` | refused: "np.log with a second positional argument", with the hint to divide by np.log(b); `math.log(x, 1.0)` raises ZeroDivisionError as CPython does (py_div_f) |
| module-path | 2026-09-27 | moving the mappings into rules | a `*` pattern reached names with a module path in the middle: `np.emath.sqrt(-1.0)` (1j in NumPy) and `np.linalg.sqrt` were `sqrt`; names no module has (`math.abs`, `math.arcsin`, `np.erf`) translated too | a translating rule names each function; a `*` pattern may only refuse; `np.emath.*`, `np.lib.scimath.*`, `np.ma.*` are refused with a hint, the rest as unknown functions |
| binding | 2026-09-27 | moving the mappings into rules | extra arguments were ignored: `np.trace(A, 1)`, `np.zeros(3, int)`, `abs(x, y)` | strict binding (above): `np.zeros(3, int)` binds the dtype, `np.trace(A, 1)` is refused ("np.trace with an offset"), any argument no parameter takes is refused |
| complex-dot-norm | 2026-09-27 | moving the mappings into rules | `np.vdot` did not conjugate; `np.dot` of complex vectors was typed real; `np.linalg.norm` of a complex array was `sqrt(sum(x * x))` | `py_cvdot` (conj(x) * y), `py_cdot`, `py_cnorm` (sqrt(re.re + im.im), as NumPy); complex matrices refused |
| like-complex | 2026-09-27 | moving the mappings into rules | `np.ones_like` and `np.full_like` of a complex array gave `czeros` | `py_cfillin(czeros(...), v)` |
| array-dtype | 2026-09-27 | moving the mappings into rules | `np.array(a, dtype=int)` of an existing array ignored the dtype; `np.asarray(a)` copied an array NumPy returns as itself; a dtype NumPy rounds to (np.float32) was dropped | a float array to int is refused (NumPy truncates), other dtypes convert or are refused; `np.asarray` of an array is the array |
| sign-nan | 2026-09-27 | moving the mappings into rules | `np.sign` of an array gave 0 for NaN | `py_signs`: NaN stays NaN |
| arctan2-type | 2026-09-27 | moving the mappings into rules | `np.arctan2` of arrays was typed as a number | typed as the array |
| fixed-crash | 2026-09-27 | moving the mappings into rules | a missing argument (`np.linspace()`) crashed the translator with KeyError; `--manifest` of a file with a syntax error crashed with NameError | a located refusal ("np.linspace without its argument `start`"); the syntax error's location is kept |
| dtypes | 2026-09-27 | check_rules.py | results were typed float where NumPy keeps int64 (np.abs, np.maximum, np.cumsum, np.full(3, 2), np.trace, ...) or gives float64 (np.trunc, np.rint of an int), so they printed as `2.0` for `2` | typed as NumPy's dtype; integer results are checked below 2^53 (py_int, py_int_arr); a Bool array into a ufunc (float16 in NumPy), an integer np.power or np.fmod, are refused |
| sizes | 2026-09-27 | check_rules.py | a negative size (`np.zeros(n)` with n < 0) gave an empty array; NumPy raises ValueError | `py_size` stops with NumPy's line; a float size is refused |
| math-scalars | 2026-09-27 | check_rules.py | `math.sqrt(v)` of an array (TypeError in Python) was elementwise; a Bool into a math function did not build | refused |
| builtins | 2026-09-27 | check_rules.py | the builtin `sum` of a matrix (Python adds the rows), `min`/`max`/`all`/`any` of a matrix, `list` of a matrix (rows shared), `abs` of a list (TypeError), `float` of an array (TypeError) translated | refused, each with the Python behavior in the hint |
| hypot | 2026-09-27 | check_rules.py | `sqrt(x*x + y*y)` overflowed above 1e154 and underflowed below 1e-154, where hypot does not; `hypot(inf, nan)` was NaN | `py_hypot` scales where the squares would overflow or underflow; tolerance 2 ulp |
| zero-signs | 2026-09-27 | check_rules.py | Viv's max and min of two zeros differ between engines, and np.maximum, np.minimum, np.fmax, np.fmin, np.max, np.min used them; np.clip propagated a NaN bound (NumPy ignores one); np.cumsum started from +0.0 | comparisons (py_zmax2, py_npmaxv, py_fmaxv); NumPy's clip loop (py_clip); cumsum starts from a[0] |
| linalg | 2026-09-27 | check_rules.py | np.linalg.cholesky and eigvalsh read the whole matrix (LAPACK reads the lower triangle); a singular matrix gave numbers where NumPy raises LinAlgError; a NaN crashed spectrum_sym; an FFT of an infinity gave NaN | py_lower_sym; the helpers stop with NumPy's line or a `py2viv:` line |
| views | 2026-09-27 | check_rules.py | a transpose, or the real or imaginary part of a complex array, is a view in NumPy and a copy in Viv: writing through `B = A.T` changed A in Python only | refused as a row is: a name bound to a view may not be written, nor the array after it |
| double-eval | 2026-09-27 | check_rules.py | a template that writes an argument twice (`np.square(f(x))`) called a def in it twice | refused: "np.square of an expression that calls `f`" |
| operators | 2026-09-27 | check_rules.py | `-a` and `a ** y` of an array a def returned did not build; arithmetic on a Python list (a TypeError, or repetition), on a shape (tuple repetition), on two Bool arrays (logical in NumPy), and of a real array with a complex number translated | `-1.0 * a` and `py_apow`; the rest refused |
| lists | 2026-09-27 | check_rules.py | a method or attribute of a Python list (`L.sum()`, AttributeError) translated | refused |
| isclose-kw | 2026-09-27 | check_rules.py | `np.isclose` and `np.allclose` ignored an unknown keyword | refused (strict binding) |
| cross | 2026-09-27 | check_rules.py | `np.cross` of vectors that are not 3-vectors gave a 3-vector (NumPy gives a number for 2-vectors) | `py_cross` stops |
| erf-gamma | 2026-09-27 | check_rules.py | `math.erf(inf)` and `math.gamma(inf)` were NaN (Viv's vmath) | `py_merf`, `py_mgamma` |
| fill | 2026-09-27 | check_rules.py | `a.fill(v)` of a number or a string (AttributeError in Python) became `fill(f, 0)`; of an int array with a float (NumPy truncates), of a Bool array with a number, with a complex value, translated | refused |
| npsum-blocks | 2026-09-27 | probe_numpy_order.py | np.sum of more than 8192 floats was one pairwise sum; NumPy sums 8192 at a time. np.sum of 8 or more -0.0 gave -0.0 (NumPy adds its result to 0.0) | py_blk: 0.0 plus each block's pairwise sum |
| transpose-order | 2026-09-27 | probe_numpy_order.py | np.sum, np.mean, np.var, np.std and np.prod of `A.T` (or of an expression holding one) summed the row-order copy; NumPy sums in memory order | the transpose's original is summed; other values that may hold a transpose's order are refused |
| augassign-alias | 2026-09-27 | the NumPy core work | `b = a; b += 1.0` (and `v += ...` of a parameter) rebound b in Viv; NumPy changes the array in place, so a and the caller saw the change in Python only; `row += 1` of a row in a loop likewise | `x[:] = py_fit(x + v, x)` where another name may hold the array; refused through a view |
| view-writes | 2026-09-27 | the NumPy core work | a row or a transpose bound to a name was checked only for stores into its array in the same scope: `r += 1` through it, a second name for the array, a def writing the module's array, were missed | views.py (above) |
| rank-3-literals | 2026-09-27 | check_rules.py | a rank-3 list literal (`np.array([[[...]]])`) translated, and the C backend does not build it | refused: allocate np.zeros((...)) and fill it |
| astype-keyword | 2026-09-27 | check_rules.py | `a.astype(dt=float)` (TypeError in NumPy) would bind the rule's parameter name | the parameter is `dtype`, as NumPy names it |
| negative-index-eval | 2026-09-28 | the np.random work | `f(x)[-1]` wrote `f(x)` twice (`f(x)[len(f(x)) - 1]`): a def with an effect ran twice | refused when the base calls a def or draws random numbers, as a slice of one already was |
| draws-are-effects | 2026-09-28 | the np.random work | a random draw is an effect: a template that writes an argument twice, a slice of an expression, and an expression hoisted before an earlier call all have to treat it as one | the helpers that draw (`py_npr_*`, `py_npg_*`, `npr_*`) count as calls of a def in all three checks |
| hoist-order | 2026-09-27 | check_rules.py | an expression hoisted above its statement (a conditional expression, a comprehension, math.sinh's argument) ran before a call Python evaluates first: `g() + math.sinh(f())` called f first | refused when the hoisted expression calls a def and another call of a def precedes it |
| same-rank-shapes | 2026-09-28 | the broadcasting work | an operator of two arrays of one rank and different shapes (`(3, 1) + (1, 3)`, `A - A.mean(axis=0, keepdims=True)`) was Viv's operator, which stops with its own error where NumPy broadcasts | the binop rules: NumPy's broadcasting, each entry one operation |
| none-index | 2026-09-28 | the broadcasting work | `a[None]` read as `a[0]` (the `None` translated as 0), and `a[None] = v` wrote `a[0]`; views.py took `a[None]` for an element | the newaxis rule; a store through `None` is refused; views.py counts `None` as a view |
| complex-ops | 2026-09-28 | check_rules.py | `/` of two arrays, one complex, did not build in Viv; a comparison of complex arrays gave one Bool (Viv compares whole arrays) | refused |
| list-compare | 2026-09-28 | check_rules.py | `L < M` of two Python lists (one Bool in Python, in order) and `L < 2.0` (TypeError) gave an array | refused |
| allclose-shapes | 2026-09-28 | the broadcasting work | np.allclose of a (2, 3) and a (3, 2) array compared them flat; NumPy raises | py_ballclose: broadcast, or the run stops with NumPy's ValueError |
| augassign-dtype | 2026-09-28 | check_rules.py | `x += 0.5` and `x /= 2` of an integer array that no other name holds rebound x as a float array; NumPy raises (it cannot cast the result into x) | refused, as for an array another name holds |
| element-array | 2026-09-28 | check_rules.py | an array stored into one element (`a[0, 0] = v` of an array v) translated and did not build; NumPy raises ValueError | refused |
| comprehension-scope | 2026-09-28 | the Python functions work | a comprehension's or a generator's variable was the program's variable of the same name: `x = 5; L = [x * 2 for x in range(3)]; print(x)` printed 2 (Python 3 prints 5) | each gets a fresh Viv name, in a scope of its own (funcs.py) |
| default-at-call | 2026-09-28 | the Python functions work | a default argument was evaluated at each call, in the caller's scope: `A = 2.0; def f(x, a=A)`, then `A = 3.0; f(1.0)` gave 3.0 (Python 2.0), and a local `A` of the caller was read | a number, a Bool, None or a module constant (np.pi) is written at the call; any other default is evaluated once, where the def runs, into a module variable (`py_dflt_f_a`) that every call leaving it out passes, so a default array is shared between the calls as in Python (the classes work replaced the refusal the Python functions work gave) |
| mixed-arith | 2026-09-28 | the Python functions work | a number that is an int on some paths and a float on others was computed as an int: `x ** 3` by squaring and typed an int (printed `1` for 1.331), `x * 2` without -0.0, `-x` as `0 - x` | with a float beside it the float operation (exact either way); otherwise refused, as its printing was; `%` and `//` too |
| mixed-spec | 2026-09-28 | the Python functions work | such a number printed with `{x:5}` or `{x:d}` took the float's form | refused (the int's and the float's forms differ) |
| append-parameter | 2026-09-28 | the Python functions work | `v.append(x)` of a parameter, or of a module list inside a def, rebound a local copy: the caller's list did not change | refused: a list is grown only where it is bound to a new list |
| sum312-numpy | 2026-09-28 | the Python functions work | under Python 3.12 the builtin sum of a list of floats was compensated also when the list could hold NumPy's float64, which CPython 3.12 adds plainly (and a generator's float sum the same) | refused in a program that uses NumPy, under 3.12 |
| tuple-print | 2026-09-28 | the Python functions work | a tuple printed as a list (`(1, 2)` as `[1, 2]`), and a list of ints and floats as floats (`[1, 2.5]` as `[1.0, 2.5]`); a store into a tuple translated (TypeError in Python) | tuples and such lists keep each entry's type (`T.elts`, a tuple `T.name == "tuple"`); printing them, and storing into them, is refused |
| param-guess | 2026-09-28 | the TheAlgorithms corpus | the first pass guessed a float for a parameter no call had typed yet, and the guess joined the ints the calls pass later (a number that is an int or a float, computed as an int) | a parameter of a def the program calls is left unknown until a call types it |
| keyword-order | 2026-09-28 | the classes work | keyword arguments were evaluated in the order of the parameters: `f(b=g(1.0), a=g(2.0))` called `g(2.0)` first | refused when they are written in another order and one of them calls a def or draws a random number |
| int-bool-join | 2026-09-28 | the classes work | a name bound to an int and later to a Bool was typed int: `x = 3; x = True; print(x)` printed `1` (Python prints True), and an `if` that set a Bool into an int variable did not build | a new name in straight-line code (`x_2`), refused inside a block, as an int and a float are |
| list-array-join | 2026-09-29 | the inference work | a Python list and a NumPy array joined to a NumPy array: a def called with both (`double([1.0, 2.0])` and `double(np.array(...))`) doubled the list, which Python repeats (`[1.0, 2.0, 1.0, 2.0]`), and printed it as an array | a list and an array never share a type: the def is translated once per argument type (infer.py), and a name bound to both is renamed in straight-line code or refused |
| dtype-join | 2026-09-29 | the inference work | arrays of two dtypes joined to a float array: a name or parameter holding an int (or Bool) array and a float array printed the ints as floats (`[1. 2.]` for `[1 2]`) and divided them as floats | as for a list and an array |
| list-item-type | 2026-09-29 | the inference work | an item stored into a Python list takes its own type: `out = [0.0] * 2; out[0] = 3` printed `3.0` (Python `3`), and `L = [1, 2]; L[0] = 2.5` wrote `2` into Viv's list of ints; `L[0] += 0.5` likewise | a store (and `+=`) of another type than the list's is refused |
| int-float-rows | 2026-09-29 | the inference work | the items of a list of rows mixing ints and floats were typed float: `[[1, 2.5]]` printed `[[1.0, 2.5]]`, and TheAlgorithms' `binary_exp_iterative` driver, whose rows mix them, "passed" with `243.0` printed for `243` (the corpus counted the number, not the text) | each column keeps its items' type (an int column computes and prints as ints; a column of both is a number of either type); printing such rows is refused |
| param-guess-arrays | 2026-09-29 | the inference work | a parameter of a def the program calls, which its body indexes, was guessed a float NumPy array on the first pass; the guess joined the list the calls pass (param-guess covered numbers only) | the calls type it: a parameter of a called def is unknown until one does |
| nested-print | 2026-09-29 | the inference work | diff.py's runner recorded a print's values in one list that a print inside another print's arguments emptied (`print(f(1), f(2))` of an f that prints): the values went to the wrong print, and a translation could be judged against the wrong numbers | each print takes the values it recorded itself (the last n, n counted per print); a line reads back as a print whose value is an int only when the text is digits |
| bool-items | 2026-09-29 | the inference work | a Bool beside numbers in a tuple, a list or a list of rows was read back as a number: `t = (True, 1.5); print(t[0])` printed `1`, `[1, 2.5, True][2]` printed `1.0`, and rows `[[True, 2.5], ...]` were a float matrix | each item keeps its type; a Bool item is read back as a Bool (`!= 0`); printing such a list, and unpacking a Bool from it, is refused |
| fstring-const | 2026-09-30 | the strings work | a string constant in an f-string field of a print was written without its conversion and spec: `print(f"{'ab'!r} {'ab':>5}|")` printed `ab ab|` (Python `'ab'    ab|`) | the field's text is Python's format of the constant, conversion and spec applied (`!a` refused) |
| diff-text | 2026-09-30 | the strings work | diff.py took a printed line whose text differed from Python's around equal numbers for `within` ("numbers agree; the text around them differs"), and a report whose numbers agreed passed bit for bit: the fstring-const translation above passed, and so did list-of-arrays below | a line must be CPython's text; only where numbers differ within a declared tolerance may the text differ (in the digits, and in the layout NumPy chooses for an array of those values) |
| list-of-arrays | 2026-09-30 | diff-text's strict comparison (check_rules.py, `list.append-statement`'s own test) | a NumPy array appended to a list, or made by a comprehension, was a row of a list of lists: `b.append(np.array([1.5, 2.0])); print(b)` printed `[[1.5, 2.0]]` (Python `[array([1.5, 2. ])]`), and `print(b[0])` printed `[1.5, 2.0]` (Python `[1.5 2. ]`) | the list is a list of NumPy arrays: a row read from it is a NumPy array (printed and computed with as NumPy's), printing the list is refused, and rows of arrays and rows of lists in one list are refused |
