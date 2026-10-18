# Numeric Python for Viv: the subset py2viv translates

py2viv reads a Python file with CPython's own `ast` module and writes Viv source.
It has one mode, and one contract: **a translated program behaves exactly as
CPython (with NumPy) runs the Python, or py2viv refuses it with a location.**
Where the Python would raise and a double would not (a zero divisor, a math
domain error, an int past 2^53), the translation calls a guard that stops the
run with the traceback's last line, as the uncaught exception ends CPython.
Operations whose last bits come from a different library (Viv's libm against the
platform's, LAPACK, BLAS, the FFT) are kept, listed in the manifest's `inexact`,
and compared bit for bit unless the program states a tolerance (below, "What is
exact"). There is no lenient mode.

For anything outside the subset py2viv stops with the Python file, line and
column, a caret under the construct, and what to write instead. It collects
every refusal in the file, so one run lists them all:

```
t.py:4:12: not numeric Python: a list comprehension with a condition (its length is known only at run time) (lifted by R17)
    return [x for x in xs if x > 0]
           ^^^^^^^^^^^^^^^^^^^^^^^^
  Count first, allocate np.zeros(count), then fill in a second loop; growable arrays are R17.
```

`(lifted by R3)` and similar name the language gap, numbered by the idioms study, whose fix would
let py2viv accept the construct.

Every mapping rule has a generated differential test (`check_rules.py`: its seed inputs,
then edge values, each argument as an expression, every keyword, an argument too many or
missing), and the examples (`examples/mapping.py`, `examples/adapters.py`,
`examples/lj.py`, `examples/mlp_grad.py`; run them with `diff.py`) run CPython with NumPy
against the Viv interpreter, Viv compiled to C, and Viv compiled to wasm. The guards,
reductions, bit operations, f-strings and several results each have a unit test that
compares the translation's output with CPython's text byte for byte
(`tests/test_py2viv.py`, `RunsInViv`, `NumPyOrder`, `Differential`).

Probed with CPython 3.11.5 and 3.12.4, NumPy 1.24.4, and the Viv of the
records-tuples branch (several results, `stop("...")`) on 2026-09-27. The
exceptions' wording is taken from the CPython that runs py2viv (evaluated there,
`HOST_ERRORS`), since CPython rewords them between versions.

## Using it

```
python3 tools/py2viv/py2viv.py prog.py > prog.viv            # translate
python3 tools/py2viv/py2viv.py prog.py -o prog.viv --manifest prog.json
python3 tools/py2viv/diff.py prog.py                        # translate, run four ways, report
python3 tools/py2viv/corpus.py posts                        # the posts/refs corpus
/opt/homebrew/bin/python3.12 tools/py2viv/corpus.py thealgorithms
python3 tools/py2viv/grad_preview.py                        # NumPy MLP gradient vs torch
python3 -m pytest tools/py2viv/tests                        # unit tests
python3 tools/py2viv/rule_engine.py --table                 # every mapping rule and its exactness
python3 tools/py2viv/check_rules.py                         # every rule's generated differential test
tools/py2viv/check_rules_equiv.sh                           # identical to the table-in-code translator, except the listed fixes
target/release/viv run prog.py                              # translate and run in the interpreter
target/release/viv check prog.py                            # translate and check, running nothing
target/release/viv prog.py -o prog.c                        # translate and compile (C, or -o prog.wasm)
```

`viv` translates a `.py` file (or a file that declares `#! lang python`) with
this translator, and every error, in every engine, is shown at the Python file,
line and column: a refusal at the construct, a compile error or a run-time
error at the Python statement the Viv line came from, with a note giving the
line of the generated Viv. It finds the translator at `VIV_PY2VIV=<path>`, or
in the repository that holds the program or the `viv` binary, and runs it with
`VIV_PYTHON` (default `python3`). The manifest's `posmap` is what makes this
work: for each Viv line, the Python span it came from, `[line, col, end_line,
end_col]` (1-based, columns in bytes), or null.

The reports from the last run of each are in `results/` (with the generated Viv
of each example), so the evidence travels with the translator.

`diff.py` finds the compiler at `VIV=<path>` (default `target/release/viv`) and
the wasm runtime at `VIV_WASM=<path>` (default `www/public/viv/viv.wasm`).

### Where the mappings live

The builtin, math and NumPy mappings in the tables below (functions, array methods and
attributes, module constants, `divmod` and `eigh` unpacking, and their refusals) are data:
`rules/builtins.toml` and `rules/numpy.toml`, with the Viv helper defs in
`rules/helpers.toml`. Each rule states its pattern, its Viv template, its exactness class
(exact, a tolerance with rtol and atol, or refused, with the evidence), a guard where one
applies, and test inputs. `RULES.md` specifies the format; the tables in "Builtins, the math
module and NumPy: the rules" below are generated from the rule files. `check_rules.py` runs
each rule's generated differential test and fails the build on a rule without passing tests
(www `npm run check`, posts/check.sh). `check_rules_equiv.sh` proves that the rules translate
every example, the posts/refs and TheAlgorithms corpora, the unit tests' inputs and a
generated corpus exactly as the table-in-code translator did, except where a listed hole was
fixed (RULES.md, "Fixed holes").

### `#! lang python`

The hook exists in the library: `viv::expand_uses_with` hands a module that
opens with `#! lang <name>` to an injected `translate(lang, module, source)`
and splices the Viv it returns. The `viv` command line does not inject a
translator yet (`lang/viv/src/main.rs` calls `expand_uses`, which refuses every
dialect), and P0 does not change the compiler. The planned wiring: `main.rs`
passes a translator that, for `lang == "python"`, runs
`python3 tools/py2viv/py2viv.py -` and returns its standard output, or its
refusals as the error. The browser build would need a Rust front end instead;
until then, translate on the command line.

## How printing works

Each Python `print(...)` becomes one Viv `print(...)`: its
text, and each value in the form Viv's `fmt` writes it.

| Python value | Viv | CPython's text, byte for byte |
|---|---|---|
| a float | `fmt(x)` (repr: the shortest decimal that reads back) | yes, `-0.0`, `inf`, `nan` included |
| an int | `fmt(n, "d")` | yes (a zero computed as `-0.0` prints `0`) |
| a bool | one print per value of the bool, writing `True` or `False` | yes (up to three bools in one print) |
| a list of floats or ints | `fmt(v)`, `fmt(v, "d")` | yes: `[1.0, 2.5]`, `[1, 2]` |
| a string, a list of strings | the value itself (Viv's print writes a string as its text and a list of strings as Python does); `repr(s)` for `!r` | yes; diff.py reads each string back as the text Python printed |
| a NumPy array | `fmt(A)`, nested brackets | no: diff.py reads the values back and prints them with NumPy |
| a complex number or vector | `"(", fmt(re(z)), ",", fmt(im(z)), ")"` | no: read back |
| `print(a, b, sep=s)` | the separator as text between them | yes |

An f-string maps to `fmt`: `{x}` and `{x!r}` are
`fmt(x)`, `{n}` of an int `fmt(n, "d")`, `{x:.3e}` `fmt(x, ".3e")`, `{x=}`
`"x=", fmt(x)`, and so on for `[width][.precision]` with type `g`, `e`, `f`, `d`;
a fill and an alignment (`{x:*>12.3f}`) format the number, then pad its text
(`fmt(fmt(x, ".3f"), "*>12")`), and a string field takes Python's string spec
(`{s:^9}`, `{s:.3}`). Everything else Python's spec can say (`=` alignment, `+`,
`,`, `%`, `n`, `#`, upper case, `.3` with no type on a number, zero padding, `d`
on a float) is refused, as is `end=` other than a newline (every Viv print ends
its line). Any text is written: a character Viv's literals do not carry raw is
an escape (`\u{7}`).

Text built for a print is written piece by piece: `print("x = " + str(x) + f", n = {n}")`
prints the literal text, and `str(x)` or `repr(x)` of a number or a list as print writes it
(`str` of a float is its repr in Python 3). A string computed another way (`"[" + ", ".join(...)
+ "]"`, a variable) is one value of the line ("Strings" below). More than three Bools in one print
are each written by `py_bstr` (True or False); up to three, the print is written once per case.

A number that is an int on one path and a float on another cannot be printed
(Python writes `3` or `3.0`); py2viv renames a variable whose type changes in
straight-line code (`y`, then `y_2`), and refuses the print otherwise.

`--print plain` writes `show x` and six-digit `print(x)` instead, for reading.

## Supported

### Statements

| Python | Viv | notes |
|---|---|---|
| `def f(a, b=2.0)` | `def f(a, b)` | a default that is a number, a Bool, None or a module constant (np.pi) is written at each call; any other default is evaluated once, where the `def` runs, into a module variable (`py_dflt_f_b`) that every call leaving it out passes, as Python shares it; keyword arguments reordered (refused when they are written in another order than the parameters and one of them calls a def); recursion |
| a def nested in a def | a module-level def `outer__inner` whose first parameters are the enclosing variables it reads, passed at each call | one level of nesting; recursion; nested defs calling each other; refused: `nonlocal`, a nested def that reads enclosing variables used as a value, a read before its `def` runs or before the variables it reads are assigned (see "Functions and lists") |
| `f = lambda x: e` (bound once in its scope) | `def f(x) { return e }`, nested as above inside a def | |
| `g(lambda x: e, ...)`, `g(f, ...)` of a def g | a module-level def passed by name; `g` calls it through its parameter | Viv specializes `g` per function passed (its compiled engines have no closures): a lambda that reads a variable of an enclosing def is refused |
| `def test_x():` | `def py_test_test_x() { ... }` and `test "test_x" { py_test_test_x() }` | emitted after the program, as pytest runs tests after the import; a failed `assert` ends the test (as in pytest) |
| `assert cond` in a test | `if not cond { assert_true(false)  return 0 }` | the condition exactly as written (`<` stays `<`) |
| `math.isclose`, `np.isclose`, `np.allclose` | `py_isclose`, `py_npisclose`, `py_allclose` | CPython's math_isclose_impl and NumPy's isclose, infinities and NaN included |
| `x = e`, `x += e` (all arithmetic ops) | same; `//=`, `%=`, `**=` spelled out | `x += e` of an array another name, a caller or a view may hold is `x[:] = py_fit(x + e, x)`: NumPy changes the array in place; with an array e, `py_inpl(py_badd(x, e), x)`, broadcast, stopping where the result would be larger than x (NumPy's ValueError); a result x's dtype cannot hold (`+= 0.5` of an int array) is refused |
| `a[i] = e`, `a[i, j] += e`, `A[i][j] = e` | `a[i, j] = e` | a chain of indices is one index |
| `a[1:-1] = v`, `A[:, j] = v`, `A[i] = v`, `a[2:] += v` | a Viv slice write | a number, or an array NumPy broadcasts into the slice's shape (`py_bset`, which stops where NumPy raises); refused through a name bound to a view, into a Python list, a float into an int array, and a value of a higher rank |
| `a[m] = v`, `a[idx] = v`, `A[:, idx] = v`, `A[I, J] = v`, `a[m] += v` | `py_mset`, `py_setax`, `py_msetax`, `py_set2` | the entries written in row-major order of the selection, so of repeated indices the last write wins; `+=` reads the selection once, adds and writes it back, so a repeated index is added to once, as NumPy; a number, or a value that broadcasts to the selection (NumPy's errors stop the run); rules index-array-store and index-arrays-store |
| `return a, b` | `return a, b` | Viv's several results (a record the compiler declares for the def); every `return` of the def gives the same number; `return g(x)` of such a def forwards them |
| `a, b = f(x)` | `a, b = f(x)` | a call of such a def anywhere else (inside an expression, as a value) is refused |
| `a, b = b, a + b`, `x[i], x[j] = x[j], x[i]` | Viv's `a, b = b, a + b`: every right side first | direct stores when no later right side reads an earlier target |
| `a, b = v` (a vector) | `a, b = v` | Viv stops unless `len(v)` is the number of names |
| `n, m = A.shape` | `n, m = shape(A)` | the rank is checked |
| `q, r = divmod(a, b)` | `q = a // b`, `r = a % b` | |
| `raise E("message")` | `stop("E: message")` | the run ends with the traceback's last line, exit status 1; a computed message keeps the exception type and the literal parts, and diff.py compares the type (the manifest's `stops`) |
| `assert c, "m"` outside a test | `if not c { stop("AssertionError: m") }` | |
| `l, g = value_and_grad(f)(x, ...)` | `l, g = value_and_grad(f, x, ...)` | `argnums=k` through a generated wrapper that puts parameter k first; tested against torch |
| `g = grad(f)(x, ...)` | `grad(f, x, ...)` | |
| `if` / `elif` / `else`, `while`, `break`, `continue`, `pass` | same | truthiness of a number is `x != 0` |
| `for i in range(n)`, `range(a, b)`, `range(a, b, s)` | `for i to n`, `for i from a to b`, `... by s` | a negative step included; a loop variable the body rebinds runs on a hidden counter |
| `for i in reversed(range(...))` | `... reversed` | |
| `for x in a` (an array) | a loop over the index | a row of a matrix is `slab(A, i)`, a copy |
| `for i, x in enumerate(a)`, `for x, y in zip(a, b)` | a loop over the index | zip stops at the shorter |
| `for x in <generator>`, `map(f, a)`, `filter(f, a)`, `enumerate(a, 1)`, `reversed(a)`; `for a, b in rows` | the generator's loops, then the body | a row of the wrong length stops with Python's ValueError; `break` under a generator of several `for` clauses is refused |
| `lst.pop()`, `lst.pop(i)`, `insert`, `extend`, `remove`, `clear`, `reverse` | `lst = py_dropv(lst, k)` and the like: the list rebound | a list nothing else holds (the proof `lst.append` needs; a parameter when every call passes a new list); IndexError and ValueError as CPython words them; `pop` also as the whole value of an assignment or a return, or as the argument of another list's `append` |
| `lst.sort(key=f, reverse=True)` | a stable merge sort of the keys, the items written back in place | a NaN key stops the run |
| `for h, n in ((0.05, 1), (1.0, 2))` | array literals indexed per pass | numbers only |
| `for name, A in (("x", np.eye(2)), ...)` | unrolled, up to 16 rows | a string constant is written where the body reads the name (the body never rebinds it); `for w in ("a", "b")` of strings only is a loop over a list of strings |
| `if __name__ == "__main__":` | its body at top level | |
| a name reused for a value of another type | a new Viv name (`y_2`) | only in straight-line code; inside an `if` or a loop it is refused |
| docstrings, comments | `#|` intent above a def; comments kept | |
| `import numpy as np`, `import math`, `from math import sqrt`, `from numpy import ...` | | |

### Expressions

| Python | Viv | notes |
|---|---|---|
| int and float literals up to 2^53 | numbers | Viv has one number type, float64 |
| `True`, `False` | `true`, `false` | stored into an array as 1 and 0 |
| `+ - * /`, unary `-` | same | tested |
| `**` | ints: `py_ipow` (by squaring, exact, stops at 2^53); floats: `py_fpow` | float_pow's errors: ZeroDivisionError for 0.0 to a negative power, OverflowError; a negative base to a fractional power (complex in Python) stops |
| `%`, `//` on numbers | `py_mod_*`, `py_floordiv_*`: CPython's float_divmod (fmod, then the sign and rounding fixes) | ZeroDivisionError on a zero divisor; `-0.0` where CPython gives it |
| `a % b`, `a // b` for a vector a and a number b | `py_mod`, `py_floordiv`: NumPy's npy_divmod per entry | a zero divisor gives NaN or `a / b` (float) and 0 (int), as NumPy |
| `+ - *` of ints | `py_int(...)` around the result | stops at 2^53 (Python's int is exact beyond it, a double is not); a `-0.0` becomes 0; `a + b` of two lengths or small literals is left bare |
| `/` | `py_div_i`, `py_div_f` | ZeroDivisionError, worded as the running CPython words it; a nonzero literal divisor is left bare |
| `&`, `\|`, `^`, `~`, `<<`, `>>` on ints | `py_band`, `py_shl`, `py_shr`, `py_int(0 - x - 1)` | exact for every int below 2^53: each operand splits into a signed high part and 26 low bits that Viv's 32-bit builtins combine without loss; `<<` stops past 2^53, a negative shift raises ValueError; tested against CPython on 22 edge values (0, 2^26, 2^31, 2^32, 2^52 - 1, negatives) in three engines |
| `@` | `@`; `sum(a * b)` for two vectors | Viv refuses `@` on two vectors; tested |
| `<`, `<=`, ..., chains `a < b < c` | `a < b and b < c` | on arrays: a 0/1 mask |
| `x in (1, 2)` | `(x == 1 or x == 2)` | literal tuples and lists |
| `x in lst`, `x not in lst`, `x in a` | `py_inlist(lst, x)`; `x in a` of a NumPy array (Viv's `in` on an array is `(a == x).any()`, NumPy's) | a NaN looked for in a Python list stops the run (CPython finds it only as the same object) |
| `lst == other`, `!=` of two lists or tuples | `py_listeq`: one Bool | a list is never equal to a tuple; a NaN at one place in both stops the run |
| `lst.index(x)`, `lst.count(x)` | `py_indexv`, `py_countv` | as `in` |
| `and`, `or`, `not` | same, parenthesized for Viv's precedence | Viv's `and` and `or` share a level; `a or b` of numbers used as a value is refused (Python returns an operand); a right side that needs statements first (a generator, a conditional expression) runs under an `if` on the left, as Python short-circuits |
| `a if c else b` | `(a if c else b)`, Viv's own; an `if` before the statement when a branch needs statements first | only the branch chosen is computed, in both; `return a if c else b` becomes two returns |
| `[1.0, 2.0]`, `[[1, 2], [3, 4]]` | array literals | |
| `[0] * n`, `[v] * n`, `[True] * n` | `zeros(n)`, `py_full(n, v)` | |
| `[e for x in s if c ...]`: s a range, a vector, a list of rows, zip, enumerate, reversed; several clauses | Viv's comprehension `[e for x in s if c]` when the loops are only heads, filters and the item (Viv writes it as the same loops); otherwise one clause, no condition, a number: allocate, then a loop; else `lst = []` and `lst = append(lst, e)` | the comprehension's variables are its own (fresh Viv names); a list element grows rows of a matrix (rows of different lengths stop the run); a tuple element makes rows whose entries keep their types |
| `sum(...)`, `any(...)`, `all(...)`, `min(...)`, `max(...)`, `next(...)`, `sorted(...)` of a generator | an accumulator loop | `any`/`all` stop at the deciding item; `min`/`max` keep Python's order (the first of equal items), `default=`, ValueError when empty; `next(g, default)`, StopIteration |
| `max(xs, key=f)`, `min`, `sorted(xs, key=f, reverse=True)` | the key applied to each item (a lambda inline, a def called) | sorted is stable, and descending with equal keys in their order, as CPython's; a tuple key is lexicographic; a NaN key stops the run |
| `(lambda x: e)(a)`, `map(lambda x: e, xs)` read at once | the lambda's body with its parameters bound to the arguments | read by list, sorted, sum, min, max, any, all, next, a for loop or a comprehension |
| `list(range(...))`, `list(zip(...))`, `list(enumerate(...))` | the comprehension of the items | |
| `[[0] * m for _ in range(n)]` | `zeros(n, m)` | |
| `a[i]`, `a[i, j]`, `a[-1]` (a literal) | the same: a negative index counts from the end in Viv too | |
| `a[lo:hi]`, `a[lo:]`, `a[:hi:s]`, `A[:, j]`, `A[1:-1, 1:-1]`, `T[i, :, 2:]`, `A[i][1:]` | a Viv slice (a copy) | each bound as Python's slice.indices reads it (negative from the end, clamped: `py_lo`, `py_hi`); a plain index beside a range through `py_ix`; a literal step of 1 or more (a variable step is refused); `a[1:3][0]` is refused |
| `v[::-1]`, `v[hi:lo:-k]` of a vector | `py_rslice`: a copy walked backward | the bounds as slice.indices reads them for a negative step; a view in NumPy, so the view check applies; NumPy sums it in index order, as the copy (checked); of a matrix, refused |
| `A[i]` of a matrix (read) | `slab(A, i)`, a copy | |
| `x - y`, `A - A.mean(axis=0)`, `A / A.sum(axis=1, keepdims=True)`, `x[:, None] - y[None, :]`, `a < b` of two arrays | `py_bsub(x, y)`, `py_blt(a, b)`, ... | NumPy's broadcasting (rules broadcast and broadcast-compare): the operands expanded to the common shape, then one operation per entry; operands of one shape by their form (one array, `u[2:]` beside `u[1:-1]`, `np.roll(u, 1)` beside `u`, arithmetic of these with numbers) keep Viv's operator; shapes that do not broadcast stop with the start of NumPy's ValueError; up to four axes |
| `c[:, None]`, `c[np.newaxis, :]`, `A[:, None, :]` | `py_newax(c, [1])` | a view in NumPy: the view check applies (rule newaxis) |
| `a[a > 0]`, `a[m]`, `A[m]`, `A[:, m]` | `py_mask`, `py_maskax` | a copy; a mask of the wrong shape stops with NumPy's IndexError |
| `a[idx]`, `A[:, idx]`, `A[I, J]`, `A[i, idx]`, `a[[0, 2]]` | `py_gath`, `py_gath2`, `py_gath3` | a copy; a negative index counts from the end, one out of bounds stops with NumPy's IndexError; index arrays broadcast together |
| `np.nonzero(a)[k]`, `np.where(m)[k]`, `i, j = np.nonzero(A)`, `np.flatnonzero`, `np.argwhere`, `np.count_nonzero` | `py_nzax`, `py_nz2`, `py_nzflat`, `py_argwhere` | NaN is nonzero, -0.0 is zero; the tuple itself is refused as a value |
| a slice, a row, a transpose, `np.reshape`, `ravel`, or a def returning one, bound to a name or looped over | the copy | NumPy's is a view: refused when anything in the scope writes into the array or through the view (a store, `+=`, an in-place method, a second name, a def that writes its argument or the module's array; views.py). A slice of a rank-1 Python list is a copy in Python too |
| `a.shape`, `a.shape[k]`, `a.size`, `a.ndim`, `a.T` | `shape(a)`, `len(a)`/`shape(a)[k]`, `numel(a)`, a constant, `transpose(a)` | tested |

### Builtins, the math module and NumPy: the rules

Every call of a builtin, a math function or a NumPy function, every array method and
attribute, and every module constant is translated by a rule in `rules/builtins.toml` or
`rules/numpy.toml` (RULES.md). The tables below are generated from those files
(`python3 tools/py2viv/rule_engine.py --table`; `check_rules.py` fails when they differ), so
they are the subset. A pattern is the Python signature py2viv binds a call against: a
parameter after `/` is positional only, a parameter with a default may be omitted, and an
argument the pattern does not name (a positional argument past the last parameter, a keyword
it does not list) is refused, never dropped. "tests" counts the rule's seed inputs; from them
`check_rules.py` generates the rule's differential test (edge values, each argument as an
expression, every keyword, an argument too many or missing), run against CPython and NumPy in
the three engines, whose last run is in `results/rule_tests.txt`.

<!-- BEGIN GENERATED: python3 tools/py2viv/rule_engine.py --table (check_rules.py --write) -->
#### Strings: builtins and the methods of str and of a list of strings (24 rules)

| Python | exactness | notes | tests |
|---|---|---|---|
| `str(object=None)` | exact | tests/test_py_str.py (CPython against the three engines, byte for byte) and check_rules.py's generated cases (results/rule_tests.json); An int is written by fmt(n, "d") (Viv's str(3) is "3.0"), a Bool as True or False (Viv's str writes true), a float as its repr, a list of floats, ints or strings as Python prints it. A NumPy array, a complex number, a list mixing ints and floats, and an object are refused. | 6 |
| `repr(obj, /)` | exact | tests/test_py_str.py (CPython against the three engines, byte for byte) and check_rules.py's generated cases (results/rule_tests.json); A string in Python's quotes and escapes (Viv's repr is CPython's, with CPython 3.11's list of printable characters); a number, a Bool and a list as str writes them. | 4 |
| `format(value, format_spec=None, /)` | exact | tests/test_py_str.py (CPython against the three engines, byte for byte) and check_rules.py's generated cases (results/rule_tests.json); The spec is a literal. Numbers: [[fill]align][width][.precision] with type d, e, f or g (an int's digits for no type); strings: [[fill]align][width][.precision]; anything else is refused. | 4 |
| `chr(i, /)` | exact | tests/test_py_str.py and check_rules.py's generated cases (results/rule_tests.json); Viv's chr stops with Python's ValueError text for a code point past 0x10FFFF and refuses a surrogate; holds for: not a surrogate (Python makes a string holding one; a Viv string is UTF-8 and stops the run) | 4 |
| `ord(c, /)` | exact | tests/test_py_str.py and check_rules.py's generated cases (results/rule_tests.json) | 3 |
| `len(x, /)` | exact | tests/test_py_str.py and check_rules.py's generated cases (results/rule_tests.json); a length in code points, as Python's | 3 |
| `float(x=None, /)` | exact | CPython's float(str), step for step (correctly rounded in each engine); tests/test_py_str.py | 7 |
| `int(x=None, /, base=None)` | exact | CPython's int(str, base); tests/test_py_str.py; holds for: a value below 2^53 in magnitude (py_int stops the run past it, where Python's int stays exact) | 6 |
| `list(x=None, /)` | exact | tests/test_py_str.py and check_rules.py's generated cases (results/rule_tests.json) | 2 |
| `s.upper()`, `s.lower()` | exact | tools/strings/cases.py (CPython's full case mappings, final sigma included); tests/test_py_str.py | 2 |
| `s.isspace()`, `s.isalpha()`, `s.isalnum()`, `s.isdigit()`, `s.isdecimal()`, `s.isnumeric()`, `s.isupper()`, `s.islower()`, `s.isascii()`, `s.isprintable()` | exact | tools/strings/cases.py (CPython's predicates on every character class); tests/test_py_str.py | 5 |
| `s.strip(chars=None, /)`, `s.lstrip(chars=None, /)`, `s.rstrip(chars=None, /)` | exact | tools/strings/cases.py (every Unicode space); tests/test_py_str.py | 3 |
| `s.split(sep=None, maxsplit=-1)`, `s.rsplit(sep=None, maxsplit=-1)` | exact | tools/strings/cases.py (split on every Unicode space, with separators and maxsplit); tests/test_py_str.py | 5 |
| `s.splitlines(keepends=False)` | exact | tools/strings/cases.py (every line boundary Python's splitlines knows); tests/test_py_str.py | 2 |
| `s.join(iterable, /)` | exact | tests/test_py_str.py (lists, generators and comprehensions of strings; the characters of a string) | 2 |
| `s.find(sub, start=None, end=None, /)`, `s.rfind(sub, start=None, end=None, /)`, `s.index(sub, start=None, end=None, /)`, `s.rindex(sub, start=None, end=None, /)`, `s.count(sub, start=None, end=None, /)` | exact | tools/strings/cases.py (positions in code points, with start and end); tests/test_py_str.py | 5 |
| `s.replace(old, new, count=-1, /)` | exact | tools/strings/cases.py; tests/test_py_str.py | 3 |
| `s.startswith(prefix, start=None, end=None, /)`, `s.endswith(prefix, start=None, end=None, /)` | exact | tests/test_py_str.py | 3 |
| `s.removeprefix(prefix, /)`, `s.removesuffix(prefix, /)` | exact | tests/test_py_str.py; py_removeprefix and py_removesuffix are CPython's (an empty affix leaves the string) | 3 |
| `s.ljust(width, fillchar=None, /)`, `s.rjust(width, fillchar=None, /)`, `s.center(width, fillchar=None, /)` | exact | tools/strings/cases.py (CPython's str.center puts the odd space by its own rule); tests/test_py_str.py | 4 |
| `s.zfill(width, /)` | exact | tools/strings/cases.py; tests/test_py_str.py | 3 |
| `a.index(x, /)`, `a.count(x, /)` | exact | tests/test_py_str.py; a missing item stops with Python's ValueError, the item in repr | 2 |
| `a.pop(i=None, /)`, `a.insert(i, x, /)`, `a.extend(x, /)`, `a.remove(x, /)`, `a.clear()`, `a.reverse()`, `a.sort()` | exact | tests/test_py_str.py (CPython against the three engines); holds for: a list nothing else can hold (py2viv.py's grow_refusal proves it and refuses the list otherwise); Each rebinds the list's name to a new list (a Viv list of strings is never changed in place): pop, insert and remove copy it. | 4 |
| `a.pop(i=None, /)` | exact | tests/test_py_str.py (CPython against the three engines); holds for: a list nothing else can hold; the whole value of an assignment or a return | 3 |

#### Builtins and the math module (27 rules)

| Python | exactness | notes | tests |
|---|---|---|---|
| `abs(x, /)`, `np.abs(x, /)`, `np.absolute(x, /)`, `math.fabs(x, /)` | exact | mapping.py:33 np.abs(x), :45 math.fabs, :46 abs(-2.5): bitwise; correctly rounded in both (SUBSET.md) | 3 |
| `min(x, /, *rest)`, `max(x, /, *rest)` | exact | mapping.py:29 max(v), min(w): bitwise; py_bmax keeps Python's order (the first entry, replaced only by a greater one), so NaN and signs of zero come out as CPython's | 3 |
| `len(x, /)` | exact | a shape read | 2 |
| `float(x=None, /)`, `np.float64(x=None, /)` | exact | mapping.py:46 float(3): bitwise | 4 |
| `int(x, /)`, `math.trunc(x, /)` | exact | mapping.py:46 int(-7.9): bitwise; py_trunc rounds toward zero as Python does | 4 |
| `np.trunc(x, /)` | exact | py_nptrunc: ceil below zero, floor otherwise, so -0.5 gives -0.0 and NaN and the infinities pass through, as C's trunc, which NumPy calls | 4 |
| `round(number, ndigits=None)` | exact | mapping.py:46 round(2.5), round(-3.5): bitwise; py_round sends halves to the even neighbor | 5 |
| `np.round(a, decimals=0)`, `np.around(a, decimals=0)` | exact | py_npround: half to even, -0.0 kept (np.round(-0.4) is -0.0), SUBSET.md 'What is exact' | 3 |
| `np.rint(x, /)` | exact | as np.round of a number; of an int a float (NumPy's rint of an int is a float64) | 3 |
| `pow(base, exp, mod=None)` | exact | a rewrite: pow(a, b) is a ** b, and the operator's own class applies (an int power is exact by squaring; a float power is Viv's libm pow, which the operator records as inexact) | 2 |
| `math.pow(x, y, /)` | within rtol 2.3e-16, atol 0 | mapping.py:46 pow(2.0, 10): bitwise; Viv's libm pow, 1 ulp against the platform's (check_rules.py: math.pow(1.5, 18.0) is 1 ulp off, 1.57e-16 relative) | 2 |
| `np.power(x, y, /)` | within rtol 2.3e-16, atol 0 | mapping.py:38 np.power(p, 1.5): within 1.5e-16 relative; check_rules.py: 0.5 ** 1.5 is 1 ulp off (1.57e-16 relative) | 3 |
| `sum(x, /, start=None)` | exact | mapping.py:28: the builtin left to right over an array (py_lsum) and CPython 3.12's compensated sum over a list of floats (py_fsum312), an integer sum exact below 2^53 (py_isum); SUBSET.md 'What is exact' | 4 |
| `np.sum(a, axis=None, *, keepdims=False)` | exact | mapping.py:28 np.sum(M): bitwise; NumPy's pairwise sum (py_npsum), an integer sum exact below 2^53 (py_isum); SUBSET.md 'What is exact' | 3 |
| `all(x, /)`, `any(x, /)` | exact | mapping.py:37: bitwise | 3 |
| `np.all(a, axis=None)`, `np.any(a, axis=None)` | exact | mapping.py:37 np.all: bitwise | 2 |
| `sorted(x, /)` | exact | adapters.py:15-19 and mapping.py:47: bitwise; an adapter, since Viv's sort(a) sorts in place and returns 0; holds for: no NaN, and not 0.0 with -0.0 (py_sort stops at run time on either) | 2 |
| `np.sort(a, axis=-1)` | exact | adapters.py:15-19: bitwise; an adapter, since Viv's sort(a) sorts in place and returns 0; holds for: no NaN, and not 0.0 with -0.0 (py_sort stops at run time on either) | 1 |
| `list(x, /)` | exact | mapping.py:47: a copy | 1 |
| `np.copy(a)` | exact | mapping.py:47 np.copy(v): bitwise | 2 |
| `math.floor(x, /)`, `math.ceil(x, /)` | exact | mapping.py:45 math.floor(-2.5), math.ceil(-2.5): bitwise; py_mfloor stops where math.floor raises (NaN, infinity) | 3 |
| `math.fmod(x, y, /)` | exact | mapping.py:45 math.fmod(-7.0, 3.0): bitwise; fmod is exact in IEEE arithmetic; py_mfmod stops where math.fmod raises | 2 |
| `np.fmod(x, y, /)` | exact | fmod is exact in IEEE arithmetic; Viv's mod is C's fmod | 2 |
| `math.isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0)` | exact | a comparison, py_isclose: isclose's own formula and defaults | 3 |
| `math.isnan(x, /)`, `np.isnan(x, /)` | exact | mapping.py:41 np.isnan(x[0]): bitwise | 3 |
| `math.isinf(x, /)`, `np.isinf(x, /)` | exact | mapping.py:49 math.isinf(math.inf), np.isinf(x[0]): bitwise | 3 |
| `math.isfinite(x, /)`, `np.isfinite(x, /)` | exact | mapping.py:49 math.isfinite(1.0): bitwise | 3 |

#### NumPy functions (100 rules)

| Python | exactness | notes | tests |
|---|---|---|---|
| `np.array(object, dtype=None)` | exact | mapping.py:51 np.array of a comprehension: bitwise; a literal or a copy | 4 |
| `np.asarray(a, dtype=None)` | exact | the same array (NumPy returns its argument when it is an array of that dtype), or a copy of a list | 3 |
| `np.zeros(shape, dtype=None)`, `np.empty(shape, dtype=None)` | exact | mapping.py:18 np.zeros(3), np.zeros((2, 2)): bitwise; np.empty is zeros, which NumPy's uninitialized memory is allowed to be. | 4 |
| `np.ones(shape, dtype=None)` | exact | mapping.py:19 np.ones(3): bitwise | 3 |
| `np.full(shape, fill_value, dtype=None)` | exact | mapping.py:19 np.full(3, 2.5), np.full((2, 2), -1.0): bitwise; fill keeps the sign of a zero | 4 |
| `np.zeros_like(x, /)`, `np.empty_like(x, /)` | exact | mapping.py:23 np.zeros_like(M): bitwise | 2 |
| `np.ones_like(x, /)` | exact | mapping.py:23 np.ones_like(v): bitwise | 3 |
| `np.full_like(x, v, /)` | exact | mapping.py:23 np.full_like(v, 4.0): bitwise; fill keeps the sign of a zero | 4 |
| `np.linspace(start, stop, num=50)` | exact | mapping.py:21 x = np.linspace(-1.5, 2.0, 8): bitwise; py_linspace is NumPy 1.24's formula (k * step + start, the last entry stop, one entry for num = 1) | 3 |
| `np.arange(a, b=None, c=None, /, dtype=None)` | exact | mapping.py:20 np.arange(5), (1.0, 2.0, 0.25), (10, 0, -3): bitwise; py_arange fills as NumPy does, start + k * delta with delta = (start + step) - start | 4 |
| `np.eye(N, M=None, k=0, dtype=None)` | exact | mapping.py:22 np.eye(3), np.eye(3, k=1): bitwise; the builtin eye(n, m, k), checked against NumPy in tools/arrays/numpy_cases.py | 4 |
| `np.identity(n, dtype=None)` | exact | mapping.py:22 np.identity(2): bitwise | 2 |
| `np.diag(v, k=0)` | exact | mapping.py:24 np.diag(v), np.diag(M): bitwise; the builtin diag(v, k), checked against NumPy in tools/arrays/numpy_cases.py | 5 |
| `np.roll(a, shift, axis=None)` | exact | mapping.py:40 np.roll(x, 2), np.roll(x, -1): bitwise (rank 1) | 3 |
| `np.diff(a, n=1, axis=-1)` | exact | mapping.py:40 np.diff(x): bitwise | 2 |
| `np.cross(a, b)` | exact | mapping.py:40 np.cross(v, w): bitwise; the same products and differences NumPy forms | 2 |
| `np.transpose(a, axes=None)` | exact | mapping.py:31 np.transpose(M): bitwise; NumPy's transpose is a view of `a`; a name bound to it may not be written, nor `a` after it (py2viv.py, the row-alias check). | 2 |
| `np.max(a, axis=None, *, keepdims=False)`, `np.amax(a, axis=None, *, keepdims=False)` | exact | mapping.py:29 np.max(M), np.amax(v): bitwise; py_npmax gives NaN if any entry is NaN, as NumPy does | 2 |
| `np.min(a, axis=None, *, keepdims=False)`, `np.amin(a, axis=None, *, keepdims=False)` | exact | mapping.py:29 np.min(x), np.amin(v): bitwise; py_npmin gives NaN if any entry is NaN, as NumPy does | 2 |
| `np.maximum(x, y, /)`, `np.minimum(x, y, /)` | exact | mapping.py:35 np.maximum(x, 0.0), np.minimum(x, w[0]): bitwise; a NaN in either operand gives NaN, and +0 over -0, as NumPy (the np_maxmin hook) | 4 |
| `np.fmax(x, y, /)`, `np.fmin(x, y, /)` | exact | np.fmax and np.fmin skip a NaN (py_fmax2, py_fmaxv), and put +0 over -0, as NumPy | 3 |
| `np.trace(a, offset=0)` | exact | mapping.py:25 np.trace(M): bitwise; the builtin trace(A, k) sums the diagonal in NumPy's order (SUBSET.md 'What is exact', tools/arrays/numpy_cases.py) | 3 |
| `np.outer(a, b)` | exact | mapping.py:25 np.outer(v, w): bitwise; one product per entry | 2 |
| `np.cumsum(a, axis=None)` | exact | mapping.py:26 np.cumsum(x): bitwise; NumPy's cumsum is sequential along its axis (a probe of 2000 random vectors: all bitwise; results/axis_order.txt for matrices) | 4 |
| `np.prod(a, axis=None, *, keepdims=False)` | exact | mapping.py:26 np.prod(v): bitwise; NumPy's product is sequential (a probe of 2000 random vectors: all bitwise) | 2 |
| `np.argmax(a, axis=None, *, keepdims=False)`, `np.argmin(a, axis=None, *, keepdims=False)` | exact | mapping.py:27 np.argmax(x), np.argmin(M): bitwise | 2 |
| `np.mean(a, axis=None, *, keepdims=False)` | exact | mapping.py:28 np.mean(x): bitwise; NumPy's pairwise sum (SUBSET.md 'What is exact') | 2 |
| `np.std(a, axis=None, *, keepdims=False)`, `np.var(a, axis=None, *, keepdims=False)` | exact | mapping.py:28 np.std(x), np.var(x): bitwise; divides by n, NumPy's pairwise sums (SUBSET.md 'What is exact') | 2 |
| `np.dot(x, y, /)`, `np.matmul(x, y, /)`, `np.inner(x, y, /)`, `np.vdot(x, y, /)` | within rtol 1e-15, atol 0 | mapping.py:30 np.dot(v, w), np.dot(M, v), np.matmul(M, M): bitwise (3x3); held to mapping.py's declared rtol 1e-15; NumPy calls BLAS, which may reorder the sums. np.vdot conjugates its first argument (py_cvdot). | 4 |
| `np.linalg.norm(x, ord=None, axis=None)` | within rtol 1e-15, atol 0 | mapping.py:39 np.linalg.norm(v), np.linalg.norm(M): bitwise; held to mapping.py's declared rtol 1e-15; NumPy's norm uses a BLAS dot for a vector; of a complex array it sums the squares of the real and imaginary parts (py_cnorm). | 4 |
| `np.where(m, x=None, y=None, /)` | exact | mapping.py:36 np.where(x > 0.0, x, -x): bitwise | 3 |
| `np.clip(a, a_min, a_max)` | exact | mapping.py:35 np.clip(x, -1.0, 1.0): bitwise; NumPy's clip loop (py_clip): a NaN in x stays, a NaN bound is ignored, of equal zeros the bound's sign wins | 3 |
| `np.sign(x, /)` | exact | mapping.py:36 np.sign(x): bitwise; NaN gives NaN and -0.0 gives 0.0, as NumPy | 3 |
| `np.square(x, /)` | exact | mapping.py:36 np.square(x): bitwise; one product | 3 |
| `math.hypot(x, y, /)` | within rtol 4.5e-16, atol 0 | mapping.py:45 math.hypot(3.0, 4.0): bitwise; sqrt(x * x + y * y) rounds twice, CPython's hypot about once (the bound is 1 ulp); py_hypot scales where the squares would overflow or underflow; check_rules.py: 3.2e-16 relative at hypot(2.5, 3.5) (two roundings) | 4 |
| `np.hypot(x, y, /)` | within rtol 4.5e-16, atol 0 | sqrt(x * x + y * y) rounds twice, libm's hypot about once (the bound is 1 ulp); py_hypot scales where the squares would overflow or underflow; check_rules.py: as math.hypot | 5 |
| `math.log(x, b, /)` | within rtol 4.4e-16, atol 0 | mapping.py:42 math.log(8.0, 2.0): bitwise; CPython computes log(x) / log(b) too, each log to 1 ulp against the platform libm (2 ulp); a base of 1 raises ZeroDivisionError as CPython's division does | 3 |
| `math.log10(x, /)`, `np.log10(x, /)`, `math.log2(x, /)`, `np.log2(x, /)` | within rtol 4.5e-16, atol 0 | mapping.py:42 math.log10(1000.0) = 2.9999999999999996 (1.5e-16 relative), math.log2(0.125): bitwise; check_rules.py: log10(0.5) and log2(0.001) 2 ulp off (3e-16 relative) | 3 |
| `math.atan2(y, x, /)`, `np.arctan2(y, x, /)` | within rtol 2.2e-16, atol 0 | mapping.py:43 math.atan2(1.0, -2.0): bitwise; a libm function (1 ulp against the platform libm, SUBSET.md) | 3 |
| `np.isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)`, `np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)` | exact | mapping.py:41 np.isclose(0.1 + 0.2, 0.3), np.allclose(v, v + 1e-12): bitwise; NumPy's formula and defaults (py_npisclose, py_allclose) | 5 |
| `np.linalg.solve(a, b)` | within rtol 1e-12, atol 1e-13 | adapters.py:37 sol: bitwise (3x3); held to adapters.py's declared rtol 1e-12, atol 1e-13; Viv's LU is its own, not LAPACK's dgesv; the heat-equation post needed rtol 1e-8 after 16 solves (corpus.py POST_TOL). | 2 |
| `np.linalg.inv(a)` | within rtol 2e-15, atol 1e-15 | adapters.py:42 Ainv: within 1.5e-16 relative; check_rules.py: 2.7e-16 relative on a shifted 3x3 | 1 |
| `np.linalg.cholesky(a)` | within rtol 1e-12, atol 1e-13 | adapters.py:40 L: bitwise (3x3); held to adapters.py's declared rtol 1e-12, atol 1e-13; Viv's Cholesky is its own, not LAPACK's dpotrf. | 1 |
| `np.linalg.eigvalsh(a, UPLO='L')` | within rtol 2e-15, atol 1e-15 | adapters.py:52 np.linalg.eigvalsh(A): within 3.7e-16 relative (Jacobi, not LAPACK); check_rules.py: 5e-16 relative on the seed matrix | 1 |
| `q, r = divmod(a, b, /)` | exact | a rewrite: q, r = a // b, a % b, whose operators are exact (SUBSET.md 'What is exact') | 2 |
| `np.fft.fft(a, n=None, axis=-1, norm=None)` | within rtol 1e-15, atol 1e-15 | adapters.py:27-29 X = np.fft.fft of an 8-vector: within 3.4e-16 absolute (Viv's FFT is not pocketfft); check_rules.py: 5.1e-16 absolute where NumPy gives an exact 0 | 1 |
| `np.fft.ifft(a, n=None, axis=-1, norm=None)` | within rtol 1e-15, atol 1e-15 | adapters.py:31 y = np.fft.ifft(np.fft.fft(x)): within 1.1e-16 absolute, 2.4e-16 relative; check_rules.py: as np.fft.fft | 1 |
| `np.real(val)` | exact | adapters.py:27 X.real: the parts are copied, not computed; NumPy's real part of an array is a view (a name bound to it may not be written: the row-alias check); of a real array it is the array itself. | 4 |
| `np.imag(val)` | exact | adapters.py:28 X.imag: the parts are copied, not computed | 2 |
| `np.conj(x, /)`, `np.conjugate(x, /)` | exact | a sign flip; of a real array a copy (NumPy returns a new array) | 3 |
| `np.sqrt(x, /, *out)`, `math.sqrt(x, /)` | exact | mapping.py:32 np.sqrt(p), :42 math.sqrt(2.0): bitwise; sqrt is correctly rounded in IEEE arithmetic | 3 |
| `np.exp(x, /, *out)`, `math.exp(x, /)` | within rtol 1.6e-16, atol 0 | mapping.py:32 np.exp(x): within 1.6e-16 relative; :42 math.exp(1.5): bitwise | 2 |
| `np.log(x, /, *out)`, `math.log(x, /)` | within rtol 2e-16, atol 0 | mapping.py:32 np.log(p): within 2e-16 relative; :42 math.log(10.0): bitwise | 2 |
| `np.sin(x, /, *out)`, `math.sin(x, /)` | within rtol 2.2e-16, atol 0 | mapping.py:32 np.sin(x), :43 math.sin(0.5): bitwise on 9 values; a libm function, 1 ulp against the platform libm (SUBSET.md) | 2 |
| `np.cos(x, /, *out)`, `math.cos(x, /)` | within rtol 2.2e-16, atol 0 | mapping.py:32 np.cos(x), :43 math.cos(0.5): bitwise on 9 values; a libm function, 1 ulp against the platform libm (SUBSET.md) | 2 |
| `np.tan(x, /, *out)`, `math.tan(x, /)` | within rtol 1.4e-16, atol 0 | mapping.py:32 np.tan(x): within 1.4e-16 relative; :43 math.tan(0.5): bitwise | 2 |
| `np.tanh(x, /, *out)`, `math.tanh(x, /)` | within rtol 1.2e-16, atol 0 | mapping.py:33 np.tanh(x): within 1.2e-16 relative; :44 math.tanh(0.5): bitwise | 2 |
| `np.sinh(x, /, *out)`, `math.sinh(x, /)` | within rtol 1.2e-16, atol 0 | mapping.py:33 np.sinh(x): within 1.2e-16 relative; :44 math.sinh(0.5): bitwise | 2 |
| `np.cosh(x, /, *out)`, `math.cosh(x, /)` | within rtol 1.4e-16, atol 0 | mapping.py:33 np.cosh(x): within 1.4e-16 relative; :44 math.cosh(0.5): bitwise | 2 |
| `np.arcsin(x, /, *out)`, `np.asin(x, /, *out)`, `math.asin(x, /)` | within rtol 2.1e-16, atol 0 | mapping.py:34 np.arcsin(x / 2.0): within 2.1e-16 relative; :43 math.asin(0.5): within 2.1e-16; np.asin is NumPy 2's name for np.arcsin. | 2 |
| `np.arccos(x, /, *out)`, `np.acos(x, /, *out)`, `math.acos(x, /)` | within rtol 2.1e-16, atol 0 | mapping.py:34 np.arccos(x / 2.0): within 2.1e-16 relative; :43 math.acos(0.5): within 2.1e-16; np.acos is NumPy 2's name for np.arccos. | 2 |
| `np.arctan(x, /, *out)`, `np.atan(x, /, *out)`, `math.atan(x, /)` | within rtol 2e-16, atol 0 | mapping.py:33 np.arctan(x): within 2e-16 relative; :43 math.atan(0.5): within 1.2e-16; np.atan is NumPy 2's name for np.arctan. | 2 |
| `math.erf(x, /)` | within rtol 4.5e-16, atol 0 | mapping.py:44 math.erf(0.5): within 2.1e-16 relative; check_rules.py: erf(-1.25) 2 ulp off (2.4e-16 relative) | 2 |
| `math.gamma(x, /)` | within rtol 2e-15, atol 0 | mapping.py:44 math.gamma(4.5): within 4.6e-16 relative; check_rules.py: gamma(-1.5) 1.3e-15 relative, gamma(3.0) 1.2e-15 (6.000000000000007) | 2 |
| `np.floor(x, /, *out)` | exact | mapping.py:33 np.floor(x): bitwise | 2 |
| `np.ceil(x, /, *out)` | exact | mapping.py:33 np.ceil(x): bitwise | 2 |
| `np.fabs(x, /, *out)` | exact | the abs rule's evidence (builtins.toml): correctly rounded | 2 |
| `np.add(x, y, /)`, `np.subtract(x, y, /)`, `np.multiply(x, y, /)`, `np.divide(x, y, /)`, `np.true_divide(x, y, /)` | exact | the operator (rules broadcast, and py2viv.py's for numbers); np.divide of two numbers is NumPy's float64 division, inf or NaN for a zero divisor (Viv's `/`), where Python's `/` raises; `out=`, `where=`, `dtype=` and the other ufunc keywords are refused (strict binding). | 4 |
| `np.nonzero(a)[k]` | exact | check_rules.py's generated cases: each axis of ranks 1 to 3, an axis past the last (IndexError), all zeros, NaN (nonzero) and -0.0 (zero), ints, Bools | 2 |
| `np.where(condition, /)[k]` | exact | np.where with one argument is np.nonzero (NumPy's documentation); check_rules.py's generated cases as np.nonzero's | 2 |
| `I, J = np.nonzero(a)`, `I, J = np.where(condition, /)` | exact | check_rules.py's generated cases: matrices with no, one and several nonzero entries, NaN and -0.0 | 2 |
| `np.flatnonzero(a)` | exact | check_rules.py's generated cases: ranks 1 to 3, all zeros, NaN (nonzero) and -0.0 (zero) | 2 |
| `np.count_nonzero(a, axis=None, *, keepdims=False)` | exact | check_rules.py's generated cases: ranks 1 to 3, all zeros, NaN (nonzero) and -0.0 (zero); NumPy returns a Python int | 2 |
| `np.argwhere(a)` | exact | check_rules.py's generated cases: ranks 1 to 3, all zeros, NaN and -0.0 | 2 |
| `np.sum(a, axis=None, *, keepdims=False)` | exact | check_rules.py's generated cases: every axis of ranks 1 to 3, negative axes, keepdims, the edge values; the order of each reduction checked against NumPy 1.24 on random arrays up to 30000 entries (results/axis_order.txt) | 4 |
| `np.mean(a, axis=None, *, keepdims=False)` | exact | check_rules.py's generated cases: every axis of ranks 1 to 3, negative axes, keepdims, the edge values; the order of each reduction checked against NumPy 1.24 on random arrays up to 30000 entries (results/axis_order.txt) | 3 |
| `np.var(a, axis=None, *, keepdims=False)` | exact | check_rules.py's generated cases: every axis of ranks 1 to 3, negative axes, keepdims, the edge values; the order of each reduction checked against NumPy 1.24 on random arrays up to 30000 entries (results/axis_order.txt) | 2 |
| `np.std(a, axis=None, *, keepdims=False)` | exact | check_rules.py's generated cases: every axis of ranks 1 to 3, negative axes, keepdims, the edge values; the order of each reduction checked against NumPy 1.24 on random arrays up to 30000 entries (results/axis_order.txt) | 2 |
| `np.prod(a, axis=None, *, keepdims=False)` | exact | check_rules.py's generated cases: every axis of ranks 1 to 3, negative axes, keepdims, the edge values; the order of each reduction checked against NumPy 1.24 on random arrays up to 30000 entries (results/axis_order.txt) | 2 |
| `np.max(a, axis=None, *, keepdims=False)`, `np.amax(a, axis=None, *, keepdims=False)` | exact | check_rules.py's generated cases: every axis of ranks 1 to 3, negative axes, keepdims, the edge values; the order of each reduction checked against NumPy 1.24 on random arrays up to 30000 entries (results/axis_order.txt) | 2 |
| `np.min(a, axis=None, *, keepdims=False)`, `np.amin(a, axis=None, *, keepdims=False)` | exact | check_rules.py's generated cases: every axis of ranks 1 to 3, negative axes, keepdims, the edge values; the order of each reduction checked against NumPy 1.24 on random arrays up to 30000 entries (results/axis_order.txt) | 2 |
| `np.argmax(a, axis=None, *, keepdims=False)` | exact | check_rules.py's generated cases: every axis of ranks 1 to 3, negative axes, keepdims, the edge values; the order of each reduction checked against NumPy 1.24 on random arrays up to 30000 entries (results/axis_order.txt) | 2 |
| `np.argmin(a, axis=None, *, keepdims=False)` | exact | check_rules.py's generated cases: every axis of ranks 1 to 3, negative axes, keepdims, the edge values; the order of each reduction checked against NumPy 1.24 on random arrays up to 30000 entries (results/axis_order.txt) | 1 |
| `np.reshape(a, newshape, /, order='C')` | exact | check_rules.py's generated cases: shapes of 1 to 4 lengths, -1, sizes that do not agree, int, Bool and float arrays; A view of `a` in NumPy (py2viv.py refuses it where a write could show that). A negative length (NumPy 1.24 takes any, usually -1) stands for the length that makes the sizes agree. | 3 |
| `np.ravel(a, /, order='C')` | exact | check_rules.py's generated cases; the entries in row-major order; A view of `a` in NumPy where it can be (py2viv.py refuses the translation where a write could show it). | 2 |
| `np.concatenate(seq, /, axis=0)` | exact | check_rules.py's generated cases: vectors and matrices along each axis, axis=None, ints, Bools, complex arrays, lists | 3 |
| `np.stack(seq, /, axis=0)` | exact | check_rules.py's generated cases: vectors along each axis | 2 |
| `np.vstack(seq, /)`, `np.row_stack(seq, /)` | exact | check_rules.py's generated cases: vectors (each a row) and matrices | 2 |
| `np.hstack(seq, /)` | exact | check_rules.py's generated cases: vectors end to end, matrices side by side | 2 |
| `np.column_stack(seq, /)` | exact | check_rules.py's generated cases: vectors (each a column) and matrices | 2 |
| `X, Y = np.meshgrid(x, y, /, *, indexing='xy')` | exact | check_rules.py's generated cases: both indexings, ints and floats; NumPy's meshgrid returns copies (copy=True). Two vectors only. | 2 |
| `np.random.seed(seed=None)` | exact | lib/nprandom/selftest.viv: NumPy 1.24.4's own draws, equal in the three engines; check_rules.py's np.random cases | 3 |
| `np.random.rand(d0=None, d1=None, /, *more)` | exact | lib/nprandom/selftest.viv: NumPy 1.24.4's own draws, equal in the three engines; check_rules.py's np.random cases | 3 |
| `np.random.random_sample(size=None)`, `np.random.random(size=None)`, `np.random.ranf(size=None)`, `np.random.sample(size=None)` | exact | lib/nprandom/selftest.viv: NumPy 1.24.4's own draws, equal in the three engines; check_rules.py's np.random cases | 3 |
| `np.random.uniform(low=0.0, high=1.0, size=None)` | exact | lib/nprandom/selftest.viv: NumPy 1.24.4's own draws, equal in the three engines; check_rules.py's np.random cases | 4 |
| `np.random.randint(low, high=None, size=None, dtype=None)` | exact | lib/nprandom/selftest.viv: NumPy 1.24.4's own draws, equal in the three engines; check_rules.py's np.random cases | 5 |
| `np.random.choice(a, size=None, replace=True, p=None)` | exact | lib/nprandom/selftest.viv: NumPy 1.24.4's own draws, equal in the three engines; check_rules.py's np.random cases | 6 |
| `np.random.shuffle(x, /)` | exact | lib/nprandom/selftest.viv: NumPy 1.24.4's own draws, equal in the three engines; check_rules.py's np.random cases | 2 |
| `np.random.permutation(x, /)` | exact | lib/nprandom/selftest.viv: NumPy 1.24.4's own draws, equal in the three engines; check_rules.py's np.random cases | 3 |
| `np.random.default_rng(seed=None)` | exact | lib/nprandom/selftest.viv: NumPy 1.24.4's own draws, equal in the three engines; check_rules.py's np.random cases | 4 |

#### Methods and attributes of an array (46 rules)

| Python | exactness | notes | tests |
|---|---|---|---|
| `a.sum(axis=None, *, keepdims=False)` | exact | mapping.py:48 M.sum(): bitwise; as np.sum | 2 |
| `a.max(axis=None, *, keepdims=False)`, `a.min(axis=None, *, keepdims=False)` | exact | mapping.py:48 M.max(), M.min(): bitwise; as np.max | 2 |
| `a.mean(axis=None, *, keepdims=False)` | exact | mapping.py:48 M.mean(): bitwise; as np.mean | 1 |
| `a.std(axis=None, *, keepdims=False)` | exact | mapping.py:48 x.std(): bitwise; as np.std | 1 |
| `a.var(axis=None, *, keepdims=False)` | exact | as np.var | 1 |
| `a.copy()` | exact | mapping.py:47 v.copy(): bitwise | 1 |
| `a.dot(b)` | within rtol 1e-15, atol 0 | mapping.py:48 v.dot(w): bitwise; held to mapping.py's declared rtol 1e-15 (BLAS) | 1 |
| `a.transpose()` | exact | as np.transpose; A view of `a`, as np.transpose. | 2 |
| `a.astype(dtype)` | exact | a copy | 2 |
| `a.tolist()` | exact | the same array, typed as a list for printing | 1 |
| `a.argmax(axis=None, *, keepdims=False)`, `a.argmin(axis=None, *, keepdims=False)` | exact | as np.argmax | 1 |
| `a.prod(axis=None, *, keepdims=False)` | exact | as np.prod | 2 |
| `a.cumsum(axis=None)` | exact | as np.cumsum | 3 |
| `a.pop(i=None, /)` | exact | tests/test_py_funcs.py (CPython against the three engines) and check_rules.py's generated cases (results/rule_tests.json); holds for: a list nothing else can hold (py2viv.py's grow_refusal proves it and refuses the list otherwise), of numbers of one type or of rows of one length; the whole value of an assignment or a return; O(n): the list is copied without its item (Viv has no shrink yet). | 5 |
| `a.index(x, /)`, `a.count(x, /)` | exact | tests/test_py_funcs.py (CPython against the three engines) and check_rules.py's generated cases (results/rule_tests.json); holds for: a Python list of numbers; x not NaN (the run stops: CPython finds a NaN only as the same object) | 3 |
| `a.append(x, /)` | exact | results/rule_tests.json (list.append-statement): bitwise in the three engines; holds for: a list bound only to new lists that nothing can alias (py2viv.py refuses the `[]` of any other list); its appends all ints or all floats, or all rows of one length (a new list or array, a copy, or a tuple: a list of lists shares its rows in Python); Viv's append gives a new array, so a second name would not see it; the translator proves there is none (grow_refusal) and refuses the `[]` otherwise. A row goes on as a new row of a matrix; rows of different lengths stop the run (a run-time refusal). | 3 |
| `a.pop(i=None, /)`, `a.insert(i, x, /)`, `a.extend(x, /)`, `a.remove(x, /)`, `a.clear()`, `a.reverse()` | exact | tests/test_py_funcs.py (CPython against the three engines) and check_rules.py's generated cases (results/rule_tests.json); holds for: a list nothing else can hold (py2viv.py's grow_refusal proves it and refuses the list otherwise), of numbers of one type or of rows of one length; pop, insert and remove copy the list (O(n)); extend and append grow it in place. | 6 |
| `a.sort(*, key=None, reverse=False)` | exact | tests/test_py_funcs.py (CPython against the three engines) and check_rules.py's generated cases (results/rule_tests.json); holds for: a Python list; no NaN key (py_msort stops: the order would be Timsort's); A stable merge sort of the keys: every stable sort gives the same order, so it is CPython's Timsort's; the list's items are written back in place (every name holding it sees them). | 2 |
| `a.sort()` | exact | adapters.py:19 a.sort(): bitwise; holds for: no NaN, and not 0.0 with -0.0 (py_sort_ip stops at run time on either) | 1 |
| `a.fill(v)` | exact | a store | 2 |
| `a.T` | exact | mapping.py:31 M.T: bitwise; A view of `a`, as np.transpose. | 2 |
| `a.shape` | exact | mapping.py:48 M.shape[1]: bitwise | 1 |
| `a.size` | exact | mapping.py:48 M.size: bitwise | 1 |
| `a.ndim` | exact | mapping.py:48 M.ndim: a constant from the static type | 1 |
| `a.real` | exact | adapters.py:27 X.real | 2 |
| `a.imag` | exact | adapters.py:28 X.imag | 2 |
| `a.sum(axis=None, *, keepdims=False)` | exact | check_rules.py's generated cases: every axis of ranks 1 to 3, negative axes, keepdims, the edge values; the order of each reduction checked against NumPy 1.24 on random arrays up to 30000 entries (results/axis_order.txt) | 2 |
| `a.mean(axis=None, *, keepdims=False)` | exact | check_rules.py's generated cases: every axis of ranks 1 to 3, negative axes, keepdims, the edge values; the order of each reduction checked against NumPy 1.24 on random arrays up to 30000 entries (results/axis_order.txt) | 1 |
| `a.var(axis=None, *, keepdims=False)` | exact | check_rules.py's generated cases: every axis of ranks 1 to 3, negative axes, keepdims, the edge values; the order of each reduction checked against NumPy 1.24 on random arrays up to 30000 entries (results/axis_order.txt) | 1 |
| `a.std(axis=None, *, keepdims=False)` | exact | check_rules.py's generated cases: every axis of ranks 1 to 3, negative axes, keepdims, the edge values; the order of each reduction checked against NumPy 1.24 on random arrays up to 30000 entries (results/axis_order.txt) | 1 |
| `a.prod(axis=None, *, keepdims=False)` | exact | check_rules.py's generated cases: every axis of ranks 1 to 3, negative axes, keepdims, the edge values; the order of each reduction checked against NumPy 1.24 on random arrays up to 30000 entries (results/axis_order.txt) | 1 |
| `a.max(axis=None, *, keepdims=False)` | exact | check_rules.py's generated cases: every axis of ranks 1 to 3, negative axes, keepdims, the edge values; the order of each reduction checked against NumPy 1.24 on random arrays up to 30000 entries (results/axis_order.txt) | 1 |
| `a.min(axis=None, *, keepdims=False)` | exact | check_rules.py's generated cases: every axis of ranks 1 to 3, negative axes, keepdims, the edge values; the order of each reduction checked against NumPy 1.24 on random arrays up to 30000 entries (results/axis_order.txt) | 1 |
| `a.argmax(axis=None, *, keepdims=False)` | exact | check_rules.py's generated cases: every axis of ranks 1 to 3, negative axes, keepdims, the edge values; the order of each reduction checked against NumPy 1.24 on random arrays up to 30000 entries (results/axis_order.txt) | 1 |
| `a.argmin(axis=None, *, keepdims=False)` | exact | check_rules.py's generated cases: every axis of ranks 1 to 3, negative axes, keepdims, the edge values; the order of each reduction checked against NumPy 1.24 on random arrays up to 30000 entries (results/axis_order.txt) | 1 |
| `a.reshape(shape, /, *, order='C')` | exact | as np.reshape; `a.reshape(2, 3)` is the rule a.reshape-lengths. | 2 |
| `a.reshape(d0, d1, d2=None, d3=None, /, *, order='C')` | exact | as np.reshape | 2 |
| `a.ravel(order='C')`, `a.flatten(order='C')` | exact | as np.ravel; flatten always copies | 2 |
| `g.random(size=None, dtype=None, out=None)` | exact | lib/nprandom/selftest.viv: NumPy 1.24.4's own draws, equal in the three engines; check_rules.py's np.random cases | 3 |
| `g.standard_normal(size=None, dtype=None, out=None)` | exact | lib/nprandom/selftest.viv: NumPy 1.24.4's own draws, equal in the three engines; check_rules.py's np.random cases | 3 |
| `g.normal(loc=0.0, scale=1.0, size=None)` | exact | lib/nprandom/selftest.viv: NumPy 1.24.4's own draws, equal in the three engines; check_rules.py's np.random cases | 4 |
| `g.uniform(low=0.0, high=1.0, size=None)` | exact | lib/nprandom/selftest.viv: NumPy 1.24.4's own draws, equal in the three engines; check_rules.py's np.random cases | 4 |
| `g.integers(low, high=None, size=None, dtype=None, endpoint=False)` | exact | lib/nprandom/selftest.viv: NumPy 1.24.4's own draws, equal in the three engines; check_rules.py's np.random cases | 6 |
| `g.choice(a, size=None, replace=True, p=None, axis=0, shuffle=True)` | exact | lib/nprandom/selftest.viv: NumPy 1.24.4's own draws, equal in the three engines; check_rules.py's np.random cases | 7 |
| `g.shuffle(x, axis=0)` | exact | lib/nprandom/selftest.viv: NumPy 1.24.4's own draws, equal in the three engines; check_rules.py's np.random cases | 2 |
| `g.permutation(x, axis=0)` | exact | lib/nprandom/selftest.viv: NumPy 1.24.4's own draws, equal in the three engines; check_rules.py's np.random cases | 3 |

#### Module constants (5 rules)

| Python | exactness | notes | tests |
|---|---|---|---|
| `np.pi`, `math.pi` | exact | mapping.py:45 math.pi: bitwise (the double, written by repr) | 1 |
| `np.e`, `math.e` | exact | mapping.py:45 math.e: bitwise | 1 |
| `math.tau` | exact | the double, written by repr | 1 |
| `np.inf`, `math.inf`, `np.Inf` | exact | (1.0 / 0.0) | 1 |
| `np.nan`, `math.nan`, `np.NaN` | exact | (0.0 / 0.0) | 1 |

#### Operators on two arrays (broadcasting) (2 rules)

| Python | exactness | notes | tests |
|---|---|---|---|
| `x + y`, `x - y`, `x * y`, `x / y` | exact | check_rules.py's generated cases: shapes that broadcast at ranks 1 to 3 (a length 1 on either side, a lower rank, an empty axis), shapes that do not (the run stops with the start of NumPy's ValueError), ints, Bools, complex numbers, NaN, inf and -0.0; each entry one IEEE operation, as NumPy's loop | 4 |
| `x < y`, `x <= y`, `x > y`, `x >= y`, `x == y`, `x != y` | exact | check_rules.py's generated cases, as `broadcast`; `==` and `!=` of shapes that do not broadcast stop with a py2viv line (NumPy 1.24 gives one False and warns, NumPy 2 raises) | 3 |

#### Indexing (6 rules)

| Python | exactness | notes | tests |
|---|---|---|---|
| `a[lo:hi:step]` | exact | check_rules.py: each bound omitted, 0, inside, at and past either end, negative, a variable, reversed (an empty slice); steps 1 to 3; each axis of ranks 1 to 3, beside plain and negative indices; each reduction of a slice checked against NumPy's order (study of 2026-09-27: a slice sums as its copy, blocked pairwise); Bounds are normalized as Python's slice.indices does (py_lo, py_hi): Viv's ranges need 0 <= lo <= hi <= n. A negative or variable step is refused (Viv's ranges step forward). | 4 |
| `a[None]` | exact | check_rules.py's generated cases: `None` and np.newaxis before, between and after the axes of ranks 1 to 3, beside ranges and plain indices; a view in NumPy, so the view check (views.py) refuses one a write could show; `axes` are the positions of the new axes in the result (py2viv.py counts them from the subscript); the subscript without them is `a`. | 2 |
| `a[i]`, `a[:, i]`, `a[:, :, i]` | exact | check_rules.py's generated cases: masks all true, all false, empty and of the wrong length (IndexError), integer indices repeated, negative and out of bounds (IndexError), on each axis of ranks 1 to 3, ints and Bools and floats in the array; entries are copied | 5 |
| `a[i, j]`, `a[i, j, k]` | exact | check_rules.py's generated cases: index arrays that broadcast together and that do not (IndexError), an int beside an array, negative, repeated and out-of-bounds indices, ranks 2 and 3 | 2 |
| `a[i] = v`, `a[:, i] = v`, `a[:, :, i] = v` | exact | check_rules.py's generated cases: a number, a vector of the selected length, one entry broadcast, a value of another length (NumPy raises); repeated indices (the last write wins), `+=` through repeated indices (added once), masks all true, all false and empty | 5 |
| `a[i, j] = v`, `a[i, j, k] = v` | exact | check_rules.py's generated cases: as `index-arrays`, with the value a number, a vector of the result's length, broadcast, or of another shape (NumPy raises); repeated pairs (the last write wins) | 2 |

#### Refused (25 rules)

| Python | refusal | hint | lifted by |
|---|---|---|---|
| `s.casefold(...)`, `s.title(...)`, `s.capitalize(...)`, `s.swapcase(...)`, `s.partition(...)`, `s.rpartition(...)`, `s.encode(...)`, `s.expandtabs(...)`, `s.translate(...)`, `s.maketrans(...)`, `s.format_map(...)`, `s.isidentifier(...)`, `s.istitle(...)` | str.{name} | Not in Viv's strings; write it from upper, lower, find and slices. | out of scope: casefold, title, capitalize, swapcase, partition, translate, expandtabs, bytes |
| `math.factorial(...)`, `math.comb(...)`, `math.perm(...)`, `math.gcd(...)`, `math.lcm(...)`, `math.isqrt(...)` | {name} | Integer functions are not mapped: write the loop (results above 2^53 are inexact in Viv). | an Int type (typed_array_elements.md), or an integer adapter per function |
| `isinstance(...)`, `type(...)`, `id(...)`, `hash(...)`, `getattr(...)`, `setattr(...)`, `hasattr(...)`, `eval(...)`, `exec(...)`, `globals(...)`, `locals(...)`, `vars(...)`, `dir(...)` | `{name}` | Dynamic Python is outside the numeric subset. | none planned: the numeric subset is statically typed |
| `input(...)`, `open(...)` | `{name}` | py2viv reads a file whole: open(p).read(), open(p).readlines(), `for line in open(p)`, `with open(p) as f` (read, a loop, write), Path(p).read_text(); a program's input comes from a file. | none planned for input (a program reads its data from files); open is translated where it is read or written whole (strs.py: open(p).read(), `with open(p) as f`, a loop over its lines) |
| `range(...)` | range outside a for loop | Use np.arange(n) for an array of indices. | none needed: np.arange is the array form |
| `enumerate(...)`, `zip(...)`, `reversed(...)`, `map(...)`, `filter(...)`, `iter(...)`, `next(...)` | `{name}` outside a for loop header | A for loop, a comprehension, or a generator read by sum, min, max, sorted, list and the like reads these, and next reads a generator expression; write the loop over indices. | none planned: a for header is the only place these are read |
| `divmod(...)` | divmod (a tuple) | Write a // b and a % b. | R5 |
| `tuple(...)`, `set(...)`, `frozenset(...)` | `{name}` | Viv arrays hold numbers; key by integer index, or use a dict (a Viv map) or a dict literal with constant keys (a record). | dict(...) is not a rule: dict(), dict(zip(ks, vs)) and dict(d) of a map translate (maps.py, a Viv map), dict(a=1, b=2) and dict(d) of a dict with literal keys (classes.py, a record) |
| `math.log1p(...)`, `np.log1p(...)`, `math.expm1(...)`, `np.expm1(...)` | {name} | Viv has no {short} builtin; write log(1 + x) / exp(x) - 1 if the precision loss near 0 is acceptable. | log1p and expm1 builtins in Viv (lang/viv/src/vmath) |
| `w, V = np.linalg.eigh(...)` | np.linalg.eigh's eigenvectors | Each eigenvector's sign (and a degenerate pair's basis) is the solver's choice, so Viv's Jacobi and LAPACK disagree; use np.linalg.eigvalsh for the eigenvalues. | none planned: each eigenvector's sign (and a degenerate pair's basis) is the solver's choice; np.linalg.eigvalsh gives the eigenvalues |
| `np.linalg.eigh(...)` | np.linalg.eigh outside `w, V = np.linalg.eigh(H)` | Unpack it on its own line. | R5 |
| `np.linalg.det(...)`, `np.linalg.eig(...)`, `np.linalg.svd(...)`, `np.linalg.qr(...)`, `np.linalg.lstsq(...)`, `np.linalg.matrix_rank(...)`, `np.linalg.pinv(...)` | {name} | Not mapped yet: eig_general(A, re, im) for eig, lib/svd for svd; det from an LU is planned. | adapters over eig_general (eig) and lib/svd (svd); det from an LU is planned |
| `np.append(...)`, `np.tile(...)`, `np.repeat(...)` | {name} | Allocate the result with np.zeros and fill it in a loop; joining and reshaping arrays is R10/R17. | R10 |
| `np.flatten(...)` | {name} | NumPy has no np.flatten (Python raises AttributeError); write a.flatten() or np.ravel(a). | R10 |
| `np.random*(...)` | {name} | Not translated. np.random.seed with rand, random_sample, uniform, randint, choice, shuffle and permutation are, and rng = np.random.default_rng(seed) with random, integers, uniform, standard_normal, normal, choice, shuffle and permutation. | each distribution transcribed into lib/nprandom from NumPy's source, draw for draw (family np-random-legacy, np-random-generator below) |
| `np.emath.*(...)`, `np.lib.scimath.*(...)`, `np.ma.*(...)` | `{name}` | np.emath returns complex values where the argument leaves the real domain (np.emath.sqrt(-1.0) is 1j), and np.ma masks entries; use the plain np function if the argument stays in its domain. | complex-valued results for a negative or out-of-domain argument, per function (np.emath) and masked arrays (np.ma): not planned |
| `a.squeeze(...)` | `.{name}` | Reshaping is R10; allocate the new shape and copy in a loop. | R10 |
| `a.append(...)`, `a.extend(...)`, `a.insert(...)`, `a.pop(...)`, `a.remove(...)`, `a.index(...)`, `a.count(...)` | list.{name} | py2viv changes a list (append, pop, insert, extend, remove, clear, reverse, sort) only when nothing else can hold it: bound only to new lists (`[]`, a list literal, a comprehension, list(...), sorted(...)) and read nowhere an alias could be taken (a second name, an argument, a container, a nested def); otherwise allocate the final size with np.zeros(n) and write by index. | R17 |
| `a.sort(...)` | the method `.sort` | np.sort(a) returns a sorted copy; a.sort() as a statement sorts in place. | none needed: np.sort(a) is the value form, a.sort() the statement |
| `a.append(...)`, `a.extend(...)`, `a.insert(...)`, `a.pop(...)`, `a.remove(...)`, `a.clear(...)` | list.{name} | py2viv changes a list (append, pop, insert, extend, remove, clear, reverse, sort) only when nothing else can hold it: bound only to new lists (`[]`, a list literal, a comprehension, list(...), sorted(...)) and read nowhere an alias could be taken (a second name, an argument, a container, a nested def); otherwise allocate the final size with np.zeros(n) and write by index. | R17 |
| `np.random*` | np.random | Call the function (np.random.rand(3)); np.random's functions are translated as calls, not as values. | a function value (np.random.rand passed as an argument) is not translated; call it |
| `np.nonzero(...)` | {name} outside `{name}(a)[k]` and `i, j = {name}(A)` | NumPy gives a tuple of index arrays: index it, np.nonzero(a)[0], or unpack it on its own line. | R5 |
| `np.meshgrid(...)` | np.meshgrid outside `X, Y = np.meshgrid(x, y)` | Unpack it into two names on its own line (two vectors only). | R5 |
| `np.random.randn(...)`, `np.random.standard_normal(...)`, `np.random.normal(...)` | `{name}` | NumPy's legacy normals (Marsaglia's polar method) take the platform libm's log of every pair of draws, and Viv's libm differs from macOS's in the last bit on about 7% of them, so the values cannot be promised bit for bit. Use a Generator: rng = np.random.default_rng(seed); rng.standard_normal(...) is translated. | Viv computing the platform libm's log bit for bit (a libm that matches macOS's, glibc's, ... per platform) |
| `np.random.RandomState(...)`, `np.random.Generator(...)`, `np.random.PCG64(...)`, `np.random.MT19937(...)`, `np.random.SeedSequence(...)` | `{name}` | Use np.random.seed(s) and the np.random functions, or rng = np.random.default_rng(seed). | the object forms of the generators np.random.seed and default_rng translate |
<!-- END GENERATED -->

### NumPy: calling-convention adapters

These Viv builtins write into an argument or sort in place, so a name-for-name
mapping is silently wrong (the 2026-09-24 prototype returned 0 for `np.sort` and
`np.fft.fft`). Each becomes a small def that allocates, calls, and returns.

| NumPy | today's Viv builtin | translation | tested |
|---|---|---|---|
| `b = np.sort(a)`, `sorted(a)` | `sort(a)` sorts in place, returns 0 | `py_sort`: copy, then sort the copy (stops on NaN, or 0.0 with -0.0) | bitwise |
| `a.sort()` (a statement) | | `sort(a)` | |
| `X = np.fft.fft(x)`, `np.fft.ifft` | `fft(X)` in place, on a complex array only | `py_fft_r`/`py_fft_c`: `czeros`, copy in, transform | within 3.4e-16 absolute (Viv's FFT is not pocketfft) |
| `x = np.linalg.solve(A, b)` | `lin_solve(A, b, x)` writes x | `py_solve` (`py_solve_multi` for a matrix b) | bitwise |
| `L = np.linalg.cholesky(A)` | `cholesky(A, L)` writes L | `py_cholesky` | bitwise |
| `np.linalg.inv(A)` | | `lin_solve_multi(A, I, X)` | within 1.5e-16 relative |
| `np.linalg.eigvalsh(H)` | `spectrum_sym(H, w)` | `py_eigvalsh` | within 3.7e-16 relative |

### np.random: NumPy's streams, draw for draw

A seeded program's random numbers are the numbers NumPy draws: the translation carries
NumPy's generators, transcribed into Viv as lib/nprandom from NumPy 1.24.4's source (tag
v1.24.4, commit 9315a9072b2636f75c831b4eca9f42a5f67ca2fb: `_mt19937.pyx`, `mtrand.pyx`,
`_generator.pyx`, `bit_generator.pyx`, `_bounded_integers.pyx.in`, `src/mt19937`,
`src/pcg64`, `src/distributions/distributions.c`, `ziggurat_constants.h`). Every value is
computed with integer-exact double arithmetic (32-bit words; PCG64's 128-bit state as 32-bit
words times 16-bit limbs of its multiplier), so the three engines agree and match NumPy on any
platform. `lib/nprandom/selftest.viv` holds NumPy's own draws (`make_refs.py`), and
check_rules.py runs every rule from several seeds, sizes and interleavings.

| Python | translation |
|---|---|
| `np.random.seed(s)` of an int in [0, 2**32 - 1], or of a list or vector of them | MT19937 seeded as NumPy seeds it: Knuth's recurrence for an int (`mt19937_seed`, not init_by_array), init_by_array for an array; ValueError outside the range, and for an empty array |
| `np.random.rand(d0, d1)`, `random_sample`, `random`, `ranf`, `sample` | 53-bit doubles, `(a >> 5) * 2**26 + (b >> 6)` over 2**53 |
| `np.random.uniform(low, high, size)` | `low + (high - low) * u`; OverflowError when high - low is not finite |
| `np.random.randint(low, high, size)` (int64) | masked rejection: 32-bit words while the range is below 2**32, the high word first above it; NumPy's bounds errors |
| `np.random.choice(a, size, replace, p)` | with replacement: randint, or the cdf of p (`cumsum`, divided by its last entry) searched at a draw; without: a permutation's first entries, or NumPy's unique-draw rounds with p; NumPy's checks of a and p (Kahan's sum within sqrt(eps) of 1) |
| `np.random.shuffle(x)`, `np.random.permutation(x)` | Fisher-Yates from the top index down with `random_interval`; a matrix by rows |
| `rng = np.random.default_rng(seed)` (also `from numpy.random import default_rng`) | SeedSequence's hash of the seed's 32-bit words (an int literal of any size, a list of them, or an int value below 2**53), then PCG64 (XSL-RR) |
| `rng.random`, `rng.uniform`, `rng.integers(low, high, size, endpoint=)` | as above, with Lemire's method for the ints (a 32-bit draw keeps the other half of its 64-bit output for the next one) |
| `rng.standard_normal`, `rng.normal(loc, scale)` | the 256-strip ziggurat, `loc + scale * z` |
| `rng.choice(a, size, replace, p, shuffle=)`, `rng.shuffle`, `rng.permutation` | Floyd's algorithm with the trailing shuffle (or a partial shuffle of a population above 10000), Fisher-Yates |

A size is None, an int or a pair. A Generator is the state array: it may be bound to a name,
passed to a def and returned (Python shares the object, Viv the array), and drawn from; any
other use is refused.

What stays out, and why:

- **No seed.** A legacy draw in a file that never calls `np.random.seed`, `np.random.seed()`,
  `default_rng()`: NumPy seeds from the operating system, so there is nothing to compare. A
  legacy draw that runs before the seed stops (`py2viv: a draw from np.random before
  np.random.seed`).
- **The legacy normals** (`np.random.randn`, `standard_normal`, `normal`): Marsaglia's polar
  method takes the platform libm's `log` of every accepted pair, and Viv's libm (fdlibm)
  differs from macOS's in the last bit on about 7% of those inputs, so no Viv code can promise
  NumPy's values. Use a Generator.
- **A Generator's normals where NumPy uses the platform's libm.** The ziggurat returns most
  draws from integer arithmetic and the tables; about 1 draw in 4000 falls in the base strip's
  tail, where NumPy calls `log1p`, and there the run stops (`py2viv: a normal draw reached the
  ziggurat's tail`). The wedge test compares a draw with `exp(-x*x/2)`: the translation
  decides it only when the two are more than 8 ulp apart, and stops otherwise (not seen).
- Other distributions, `RandomState`, `Generator(PCG64(...))`, other dtypes (`int32`,
  `float32`: another stream), array-valued parameters (broadcasting), a size of three or more
  dimensions, `out=`, `axis=` other than 0.
- A draw written twice by the translation: a slice or a negative index of an expression that
  draws, an argument a template writes twice, a draw inside an expression py2viv evaluates
  before the rest of its statement (a conditional expression) after another draw.

Time for 10^6 draws summed in a loop (Apple M-series, 2026-09-28): MT19937 doubles 11.2 s in
the interpreter, 0.33 s compiled to C, 0.74 s in wasm under node; PCG64 doubles 12.0 s, 0.32 s,
0.31 s; a Generator's normals 13.2 s, 0.31 s, 0.50 s.

## Functions and lists

`funcs.py` holds this part of the translator: a rewrite of the syntax tree before translation
(nested defs, `f = lambda`, lambdas passed to defs, `map` and `filter`), and the translator's
methods for comprehensions, generators, key functions, list methods and lists of rows. Each
construct is translated where the Viv computes what CPython computes, and refused otherwise:

- **Closures read at the call.** Python reads a nested def's (or a lambda's) enclosing
  variables when it runs, not when it is defined. The lifted def takes them as parameters,
  passed at each call, so a variable rebound between the def and the call is read as Python
  reads it; nothing can rebind it while the call runs (`nonlocal` is refused). An array is
  passed as the array, which Viv shares as Python does. A nested def that reads enclosing
  variables cannot be a value (Viv's compiled engines have no closures): used as one, it is
  refused. A nested def read before its `def` runs, or before a variable it reads is
  assigned (NameError in Python), is refused.
- **Function values.** A def (or a lambda lifted to one) passed to a def of the program is a
  Viv function value; the callee calls it through its parameter, and Viv's C and wasm engines
  specialize the callee per function passed (lang/viv/src/defun.rs). The defs that reach a
  parameter type its calls, and the view check (views.py) follows a call through the parameter
  to them.
- **Comprehension scopes.** A comprehension's or generator's variables, and a lambda's
  parameters, are local to it in Python 3: each gets a fresh Viv name (they overwrote the
  program's variables of the same name before 2026-09-28).
- **Order and short-circuit.** A generator's items are produced in Python's order; `any`,
  `all` and `next` stop at the deciding item, through every `for` clause; a generator's first
  iterable is evaluated once, before its loops, the others inside them.
- **Sums.** The builtin `sum` of ints is exact (checked below 2^53), of floats left to right
  under Python 3.11, compensated as CPython 3.12 does under 3.12 (CPython 3.12 compensates only
  exact Python floats; in a program that uses NumPy, whose float64 items it adds plainly, a
  float sum is refused under 3.12).
- **Sorting.** `sorted` and `list.sort` with `key=` or `reverse=` sort a permutation of the
  items by their keys with a stable merge sort. For keys that `<` orders consistently every
  stable sort gives the same order, so it is CPython's Timsort's (reverse=True keeps equal keys
  in their order, as CPython does). A NaN key makes `<` inconsistent and the run stops.
- **Identity.** A Python list compares its items with `is` before `==`, so a NaN is found in a
  list, counted, or equal to another list's NaN only as the same object. py2viv cannot tell
  objects apart: `in`, `index`, `count`, `remove` of a NaN, and `==` of lists that hold a NaN
  at one place, stop the run.
- **Lists of rows share rows.** A list of lists holds references to its rows, and Viv's matrix
  holds copies: a row another name can reach may go into a list only when the program never
  writes into a row of any list or matrix. Tuples cannot change, so a list of tuples always
  may; a tuple's entries keep their types (int or float), and printing a tuple (with
  parentheses in Python) is refused.
- **Growing and shrinking.** The list methods rebind the list's name to Viv's new array
  (`append` and `extend` grow it in place; `pop`, `insert` and
  `remove` copy it, O(n)). That is exact only for a list nothing else holds: py2viv proves it
  as for `append` (`grow_refusal`), and refuses a list bound in another scope, and a
  parameter's list unless every call passes a new list (see "Typing: defs typed by their
  calls").
- **Items keep their types.** A Python list holds each item as it is: `out = [0.0] * n;
  out[0] = 3` holds an int, which prints as `3`. Viv's array holds one kind of number, so a
  store (or `+=`) of another type than the list's is refused. Rows mixing ints and floats
  (`[[3, 0.5], [4, 1.5]]`) keep a type per column: an int column computes and prints as ints,
  a column of both is a number of either type; printing such rows is refused.
- **Ints and floats.** A number that is an int on some paths and a float on others is typed so:
  with a float beside it, arithmetic gives a float either way; where Python's int and float
  results differ (the sign of a zero, a power, `//`, the 2^53 check), it is refused, as is
  printing it. `max(3, 2.5)` is such a number.

## Typing: defs typed by their calls

Python's defs take any type; a Viv def is monomorphic once compiled, and the translation of a body
depends on its parameters' Python types: an int and a float compute and print differently, a
Python list and a NumPy array are different values (`v * 2` repeats one and doubles the other,
`b = v[:]` copies one and views the other), an int array divides and prints as itself.
`infer.py` holds this part of the translator.

- **Parameters from every call.** A parameter's type is the type of the arguments the calls pass,
  found over the translator's passes (a fixed point over the call graph, recursion included), not
  a guess from the body. A def the program calls is not guessed at all until a call types it.
- **One def per argument types.** A def called with arguments of different types is translated
  once per distinct tuple of argument types: `scale(3, 4)` and `scale(2.5, 2)` call `scale` (exact
  integer arithmetic, printed `12`) and `scale__t2` (floats), `norm2([1.0, 2.0])` and
  `norm2(np.array(...))` a list clone and an array clone. Each clone's return type is its own,
  tuple returns included, and a call inside a clone calls the clone of its own argument types, so
  chains (`twice` calling `show`), mutual recursion and recursion into another type (`half(x / 2)`)
  are each typed exactly. At most 8 clones per def; past that the calls are refused. A def is
  cloned only where that is Python's behavior: a plain module-level def, not a method, not a test,
  not differentiated by `grad`, not used as a value, not passed a function, not one that changes
  an object passed to it. Those keep one def, and a call that passes another type is refused.
- **What stays one type.** A value whose type is decided at run time is a number of either type:
  `vals = [1, 2.5]; f(vals[i])` passes an int or a float, and so does `power(1.5, 0)`, whose
  `return 1` is an int where the other return is a float. Their arithmetic and printing are
  refused where Python's int and float differ.
- **List parameters a def changes.** `v.append(x)`, `v.pop()`, `v.insert`, `v.extend` of a
  parameter are translated when every call passes a new list that no other name holds: a literal,
  a comprehension, `list(...)`, `sorted(...)`, `[x] * n`, a slice of a list, a copy, or what a def
  returns when every `return` of it gives such a list (a greatest fixed point over calls and
  returns, rechecked on every pass). The caller cannot see the change, so Viv rebinding the
  parameter alone is what Python does: merge sort's `merge(merge_sort(c[:mid]),
  merge_sort(c[mid:]))`, popping both lists, translates. A call that passes a name the caller
  still holds, a module list, a default (Python shares it between the calls) or a list a def
  returns that may be one of those, is refused. The items added must be of the list's type.
- **Annotations are evidence.** (Except `int` and `Optional[...]` on a module-level def's
  parameters and result and on variables, which are translated as Viv types: "Ints, enums and
  optionals" below.) `def f(n: int, v: list[float], A: np.ndarray)` and Pythran's
  `#pythran export f(float64[], int)` state types. A call whose argument contradicts the annotation
  is refused (`int` given a float, `list[...]` given a NumPy array, `np.ndarray` given a list or a
  number; `float` takes an int, as PEP 484's numeric tower says). A def never called is typed by
  its annotations, else by its body's use (an int where it bounds a `range` or indexes, a
  sequence where it is iterated or indexed, a float otherwise); the translation of such a def is
  never run.
- **Unknown because of a refusal.** A statement py2viv refuses binds nothing it can type, a def
  whose body has a refusal returns nothing it can type, and a parameter only ever passed such
  values, or of a def called only from refused code, has no type. A use of such a value
  (`ax.set_xlabel(...)` after `fig, ax = plt.subplots()` was refused) is reported as a cascade:
  "the method `.set_xlabel` of a value of unknown type (it comes from a construct refused
  elsewhere)". A value whose type is unknown for another reason keeps its own refusal.

The generated tests (`check_rules.py`, pseudo-rules `infer.clone`, `infer.list`,
`infer.columns`, `infer.hints`) call a def at two places with a value of each pair of kinds and
return, double, add, negate or print it; change list parameters with every form of new list and
with lists the caller still holds; read, loop over, store into and print rows mixing ints and
floats; and call annotated parameters with every kind of value. They demand the printed text be
CPython's byte for byte. `tests/test_py_infer.py` runs longer programs (merge sort, recursion
across clones, tuple returns per clone) and the refusals.

## Ints, enums and optionals

`ints.py` holds this part of the translator: Viv's `Int`, enums and optionals,
each translated exactly or refused. A program that annotates nothing
`int` or `Optional[...]` and defines no `enum.Enum` class is translated as before, byte for byte.

**Ints.** A parameter or the result of a module-level def annotated `int`, and a variable
annotated `int`, is a Viv `Int`: signed 64-bit, every operation checked.

| Python | Viv | notes |
|---|---|---|
| `def f(n: int) -> int:` | `def f(n: Int) -> Int {` | a method's annotations and a def `grad` differentiates are not followed (their ints stay doubles) |
| `x: int = e` | `x: Int = e`, at every binding of `x` | |
| `a + b`, `a - b`, `a * b` of Ints | the same operators on Ints | a result past 64 bits stops the run in all three engines ("integer overflow: ... does not fit in an Int"); Python's int does not overflow, so the stop is a refusal at run time |
| `a // b`, `a % b` of Ints | `a // b`, `a % b`, or `py_ifloordiv(a, b)`, `py_imod(a, b)` when the divisor is not a nonzero literal | Viv's Int `//` floors and `%` takes the divisor's sign, Python's rules; a zero divisor stops with Python's ZeroDivisionError line |
| `a / b` of Ints | `a / b`, or `py_idiv(a, b)` | a number: each Int converted exactly (Viv stops past 2^53, where Python rounds the exact quotient) |
| `n ** k`, k a literal 0 or more | `n ^ k` | another exponent: the Ints as numbers (below) |
| `-n`, `abs(n)`, `min(a, b)`, `max(a, b)`, `round(n)`, `int(n)` | the Int forms | |
| `float(n)` | `float(n)` | the nearest double, Python's rule |
| `n < x`, `n == x` with a number x | `n < x` | Viv compares an Int with a number exactly, as Python does |
| `print(n)`, `str(n)`, `f"{n:8d}"`, `f"{n:>8}"`, `f"{n:.3f}"` | `fmt(n, "d")`, `fmt(n, "8d")`, ... | the digits of every Int; a float spec converts exactly or stops |
| `if n:` | `if n != 0` | |
| `x = e`, `return e`, `f(e)` where an Int is declared and `e` is a Python int kept as a double | `int(e)`, or the literal itself | exact: py2viv keeps such a double integral and below 2^53 (its guards stop otherwise) |
| an int literal past 2^53 | the literal | only where an Int is bound (a declared variable, parameter or result) or beside an Int in `+ - * // %`; elsewhere refused, as before |

An Int anywhere else (beside a float, in a NumPy call, as an index or a range bound, in a list,
a conditional expression or a tuple) is converted to a number exactly, `(n + 0.0)`, or the run
stops (past 2^53): from there the translation is the one py2viv makes for a Python int kept as a
double, which stops past 2^53 too. A variable or a def's result that is an Int on one path and a
Python int kept as a double on another is an Int (the double is converted where it is bound).
Unannotated ints keep their translation.

A Bool passed for a parameter annotated `int` (Python's True is an int that prints `True`) gets
the def's clone for a Bool (infer.py), whose annotations are not followed: the translation py2viv
made before Ints.

Refused: a float, a string or a list where `int` is declared ("`f` called with a float for `n`,
which is annotated `int`", "a float where `x` is annotated `int`"); `return None` from
a def annotated `-> int`, and such a def that can reach its end without `return` (Python
returns None there); a loop variable or an unpacked value bound to a name that holds an Int
(py2viv converts what `=` stores, not those); an int-annotated def used as a value.

**Enums.** A class deriving from `enum.Enum` (`import enum` and `class C(enum.Enum)`, or `from
enum import Enum`) whose body is a docstring and assignments of distinct int or str constants to
names is a Viv enum without payloads.

| Python | Viv |
|---|---|
| `class Color(Enum): RED = 1; GREEN = 2` | `enum Color { RED, GREEN }` |
| `Color.RED` | `Color.RED` |
| `a == b`, `a is b`, `a != b`, `a is not b` of members of one enum | `a == b`, `a != b` |
| `print(c)`, `str(c)`, `f"{c}"` | `"Color." + str(c)`: Python's text `Color.RED` (Viv's str of an enum is the variant's name) |
| `c.name` | `str(c)` |
| `def f(c: Color)`, `return c`, `c = Color.RED`, `def f(c=Color.RED)` | the member passed, returned, bound, written at each call |

Refused, with the reason: `.value` (py2viv does not carry the values), `auto()`, equal values
(an alias), a member looked up by value or by name (`Color(1)`, `Color["RED"]`), the class used
as a value (iterated, `len`, passed), methods or any other statement in the body, a member name
with a leading underscore or one that is a Viv keyword, `IntEnum`, `Flag`, `IntFlag` and
`StrEnum`, a member in arithmetic, in a list, a dict, a tuple, a conditional expression or a
record (Viv's arrays, maps and records do not hold enums yet), `<` between members (Python
raises), `==` of a member and another kind of value or of members of two enums, `repr` of a
member and a format spec on it, its truth value.

**Optionals.** `Optional[T]`, `T | None` and `Union[T, None]` on a parameter, a def's result or
a variable, with T `int`, `float`, `str`, `bool` or an enum class, is Viv's `T?` (`Int?`, `Num?`,
`Str?`, `Bool?`, `Color?`).

| Python | Viv |
|---|---|
| `def f(x: Optional[int] = None) -> Optional[int]:` | `def f(x: Int?) -> Int? {`; a call without `x` passes `None` |
| `None` bound, passed or returned | `None` |
| a T value bound, passed or returned where `T?` is declared | `Some(v)` |
| reaching the end of a def whose result is optional | `return None`, as Python returns None |
| `if x is None: A else: B` | `match x { None => { A } Some(v) => { B } }`, with x read as v in B |
| `if x is not None: B else: A`, and either without an else, and `elif x is None:` | the same match |
| `x = e` where x is known not to be None | `x = Some(e)`, stored into the optional |

Refused: an optional in arithmetic, printed, compared, in a list or passed where its type is not
declared; `x is None` used as a value or beside other conditions (`if x is not None and x > 2`);
a narrowing that continues after its `if` (`if x is None: return 0` then `x + 1`: write the use
in the else); in the branch where x is not None, a read of x after a store to it and a store to
it inside a loop; an int where `Optional[float]` is declared (Python keeps the int, which prints
`3` and not `3.0`).

`tests/test_py_int_enum.py` runs the programs in CPython and the three engines (enums, Ints past
2^53 through `fact(20)` and `3037000499 ** 2`, floor division and remainder for every sign, the
three ZeroDivisionErrors, optionals of ints, floats, strings and enums, recursion), checks that
an Int past 64 bits stops every engine, and checks each refusal above.

## Classes, dataclasses, named tuples and dicts

A Viv record is a value: copying it copies its numbers, and the copy's array
fields name the same arrays. A Python object is a reference. They agree exactly while no field is rebound after the object is made:
a copy and a second reference then read the same numbers, and a write into an array field goes
through the one array in both. `classes.py` translates (no compiler change was needed; records
run in all three engines):

| Python | Viv | notes |
|---|---|---|
| `class C:` with `__init__` setting `self.f` | `record C { f: Num, ... }` and `def C(...)`, whose body is `__init__`'s with `self.f` as the local `self__f`, returning `C { f: self__f, ... }` | every attribute first set at the top level of `__init__`, before it is read; `self` not used otherwise in `__init__` (the object exists when it ends); fields are numbers, Bools, arrays, lists and other records; the records are declared at the top in dependency order |
| `@dataclass` (`frozen=`, `repr=`, `eq=`, `slots=`), `class P(NamedTuple)`, `P = namedtuple("P", ["x", "y"], defaults=[...])` | the same, `__init__` generated from the fields | defaults must be number literals (Python evaluates them once; a list or array default raises in a dataclass); `__post_init__` runs at the end of the generated `__init__` (its own attributes become fields no repr prints); `field(...)` and an own `__init__` are refused |
| `def m(self, a)`, `obj.m(x)` | `def C__m(self: C, a)`, `C__m(obj, x)` | a method named nowhere in the program is not translated (Python never runs it) |
| `@property`, `@staticmethod`, `@classmethod` (`cls(...)` builds), class attributes that are number literals | a def of `self` called where the attribute is read; a def without `self`; `C__K` module variables | |
| `p.x`, `p[0]` of a named tuple, `x, y = p`, `p._replace(x=1)`, `dataclasses.replace(p, x=1)` | a field read, a new record with the other fields copied | |
| `print(p)`, `f"{p}"` of a dataclass or named tuple, `print(d)` of a dict | the text CPython writes: `P(x=1.5, y=2)`, `{'a': 1, 'b': [0.5, 2.0]}`, an array field as its repr | a plain class prints through a `__str__`/`__repr__` that returns one f-string (inlined); without one it is refused (Python prints the address) |
| `{"alpha": 1.5, "n": 3}`, `dict(dt=0.1)`, `{1: 2.0}`; `d["k"]`, `d[1]`, `d.get("k", v)`, `"k" in d`, `len(d)`, `d.copy()`, `dict(d)` | a record per key tuple (in insertion order), a field read; `in` and `len` known here | keys are string or int constants; a key the dict lacks is refused (KeyError) |
| `for k, v in d.items()`, `for k in d`, `for v in d.values()` | one copy of the body per key, the key a constant | no `break`/`continue`; up to 32 keys |
| `obj.n += 1`, `self.x = v`, `d["k"] = v` (a field rebound) | `obj = C { n: obj.n + 1, x: obj.x, ... }` | the class (or dict shape) is then "rebound": see below |
| `obj.v += a`, `obj.v[i] = x` of an array field | `obj.v[:] = py_fit(obj.v + a, obj.v)`, `obj.v[i] = x` | in place, as NumPy does; every copy of the record names the same array |
| a method or def that rebinds a field of `self` or of an object parameter | it returns the object (or objects) and then its own values, and each call rebinds the caller's variable (`c, m = C__bump(c, 4.0)`) | the call stands as a statement, the whole right side of an assignment, or a `return`, and the object passed is a variable of the caller |

For a class (or dict shape) whose fields are rebound anywhere, py2viv refuses every use that would
let two names hold one object: `q = p`, `q = p if c else r`, `return p` of a parameter or a module
variable, the object in a field, a list or a dict, the same object passed twice to a def that
changes it, `a.merge(a)`, a module variable changed inside a def, and a module variable changed
through a call while some def reads it as a module variable (the def would see it as it was before
the call). The views check (views.py) follows objects too: `obj.v[1:3]` is a view of `obj`, a
method call is a call of `C__m(obj, ...)`, `C(a)` keeps `a`, and a transpose kept in a field is
summed in memory order or refused.

The generated tests (`check_rules.py`, pseudo-rules `objects.class`, `objects.dataclass`,
`objects.frozen`, `objects.NamedTuple`, `objects.namedtuple`, `objects.dict`) put every value kind
in a field (ints, 0, 2**52, -0.0, 1e-300, NaN, the infinities, Bools, int and float arrays, a
matrix, a list) and read it (a field, a method, a property, printed, in an f-string), write into
an array field through a copy and through a def, rebind the field in place, through a def,
through a method and through a second name (refused), replace, unpack and index a named tuple,
and use a dict's `len`, `in`, `get` and loops; `tests/test_py_classes.py` runs longer programs and
the refusals.

## Dicts with keys computed at run time: Viv maps

A dict whose keys are computed at run time is a Viv map: numbers or strings
under number or string keys, one kind of key and one kind of value per dict, in insertion order,
one handle changed in place, which is what a Python dict of numbers or strings is. A dict whose
keys are both numbers and strings (or whose values are) is refused. `maps.py` translates it (a mixin of the translator, before `classes.py`, which keeps
the dicts whose keys are all string or int constants read by constant keys):

| Python | Viv | notes |
|---|---|---|
| `{}`, `dict()`, `OrderedDict()` | `map()` | |
| `{1: 2.5, k: v}` with a computed or float key, or a literal of int keys whose variable is used as a map (a computed key, a new key, `del`, `pop`, ...) | `py_m1 = map()` then `py_m1[1] = 2.5`, ... before the statement | a literal that calls a def is built before the rest of its statement, so it must be the whole value of its statement |
| `{k: v for ...}` | Viv's map comprehension, `{k: v for ...}`, when its loops are plain heads and filters (`for c in s`, `for i, c in enumerate(s)`, `for i to n`, `if c`); otherwise a loop filling a map | Python's order: each key, then its value; a key seen again keeps its place and takes the later value |
| `dict(zip(ks, vs))` | a loop filling a map | |
| `d["k"]`, `d["k"] = v` of a map | `py_mget(d, "k")`, `d["k"] = v` | a constant string key of a dict that is a map (a dict read or grown by computed keys) |
| `defaultdict(int)`, `defaultdict(float)`, `defaultdict(lambda: 0)` | `map()`; `d[k]` is `py_mdefault(d, k, 0)`, which stores the default first | `collections.defaultdict` too; any other default factory (a list) is refused |
| `Counter()`, `Counter(xs)` | a loop counting into a map; `c[k]` is `get(c, k, 0)` | |
| `d[k] = v` | `d[k] = v`, or `py_mset(d, k, v)` when the key may be a float | ints are stored `+ 0.0` (an int computed as -0.0 prints `0`); a NaN key stops the run with a `py2viv:` message (Python stores it and finds it again only through the same float object; a map holds no NaN key) |
| `d[k]` | `py_mget(d, k)` | a missing key stops with `KeyError` (the traceback's last line starts so; the key after it is compared by type); a map of strings has its own copy of each helper, `py_mget_sn` (string keys, number values), `_ns`, `_ss`, since a Viv def's map parameter has one kind of key and of value |
| `d[k] += v` (any operator) | the key once, then `d[k] = py_mget(d, k) + v` | a defaultdict reads through `py_mdefault` |
| `k in d`, `k not in d`, `k in d.keys()`, `len(d)`, `if d:` | `has(d, k)`, `not has(d, k)`, `len(d)`, `len(d) > 0` | |
| `del d[k]`, `d.pop(k)`, `d.pop(k, x)`, `d.get(k, x)`, `d.setdefault(k, x)`, `d.update(e)`, `d.clear()`, `d.copy()`, `dict(d)` | `py_mpop(d, k)` (KeyError), `remove(d, k, x)`, `get(d, k, x)`, `py_msetdefault`, `py_mupdate`, `clear(d)`, `copy(d)` | `dict(d)` of a defaultdict or a Counter is a plain dict |
| `d.keys()`, `d.values()`, `list(d)`, `sorted(d)`, `sum(d.values())`, `max(d)`, ... | `keys(d)`, `values(d)`: new vectors in insertion order, which the list rules read | a builtin that iterates a dict iterates its keys |
| `for k in d`, `for k in d.keys()`, `for v in d.values()`, `for k, v in d.items()`, `enumerate(d)`, `zip(d, ...)` | a loop over `keys(d)` (a snapshot); `v` is `d[k]` read at each step | a body that may change the size of the dict checks it at each step and after the last key, and stops with Python's `RuntimeError: dictionary changed size during iteration`; a body that may both add and remove keys is refused (Python tells a same-sized change from none, and iterates new keys) |
| a comprehension over `d`, `d.keys()`, `d.values()`, `d.items()` | over `keys(d)`, `values(d)`, or both zipped | refused when it may change a dict |
| `print(d)`, `f"{d}"` | `print(fmt(d, "d", ""))`: each key and value as Python writes an int or a float | printing a dict whose keys (or values) are ints on some paths and floats on others is refused (Python keeps the first key stored, 1 or 1.0) |

Types: a map's key and value types are learned from its stores over the translator's passes, as
a record's field types are; one creating expression is one map type (`T("map")`), which a
parameter, a return and a record field carry. A dict is a handle, so an alias, a def that
stores into its parameter, a module dict a def changes, and a dict in an object's field
(`self.counts[k] += 1`) all translate: Viv's map is shared as Python's dict is.

Dicts as arguments (maps.py with infer.py): a parameter a dict is passed to is typed by its
calls, as every parameter is (see "Typing: defs typed by their calls"). Every dict the calls
pass for one parameter becomes one map type, as dicts that meet in one variable do, before the
call picks its clone; so a def is cloned per kind of argument (`total(d)` and `total([1, 2])` are
a map clone and a list clone, each translated for its own type) and never per dict. What a def's
clones store into a map lands in that one map type: a def cloned for an int and for a float
that stores its argument into a dict it makes (or into a dict passed to it) gives a dict whose
values are ints on some paths and floats on others, whose printing and arithmetic are refused. A
dict and a defaultdict or a Counter for one parameter are refused, as in one variable. A dict
annotation is evidence as the others are: `m: dict[int, float]` (`Dict`, `Mapping`,
`defaultdict[...]`, `Counter[...]`) given something that is not a dict, given a dict whose keys or
values contradict it (floats under `int`), or a dict given to a parameter annotated `list[...]`,
`int` or `np.ndarray`, is refused. A def never called with a parameter used only as a dict is used
(a dict method, `k in m`) or annotated as one gets no guessed type.

The generated tests (`check_rules.py`, pseudo-rules `maps.dict`, `maps.literal`,
`maps.comprehension`, `maps.defaultdict`, `maps.Counter`) put every key kind (ints, 0, 2**52,
2**53 - 1, floats, -0.0, 0.0, 1e-300, 5e-324, the infinities, np.float64, NaN) through every use:
store and read, `len`, `in`, `not in`, `get`, a second key and the keys and values, a missing key
(KeyError), `+=`, `pop`, `setdefault`, `del` then a new store (the key goes last), a loop over the
items, a loop that adds or removes a key (RuntimeError), printing and f-strings; and 0.0 beside
-0.0 and 1 beside 1.0 (one key, the first kept). `tests/test_py_maps.py` runs longer programs in
the three engines against CPython, and the refusals. The pseudo-rule `infer.maps` calls a def
with a dict made each way (int and float keys) at one site and each kind of value at another, for
bodies that take its length, loop over it, sum what they iterate, return it, store into it
through the clone's parameter, and a parameter annotated `dict[int, float]`.

## Strings

Viv has strings: immutable UTF-8 values indexed by code point, with Python's
`split`, `strip`, `find`, `replace`, `repr`, `float` and `int` parsing, and CPython 3.11's Unicode
tables, in all three engines. `strs.py` (a mixin of the translator) and `rules/strings.toml` map
Python's `str` and lists of `str` onto them. The builtins and the methods are rules (the tables
above, "Strings: builtins and the methods of str and of a list of strings"); the rest is here.

| Python | Viv | notes |
|---|---|---|
| `"..."`, `'...'`, `"""..."""` | a Viv literal | any text: `\\`, `"`, `\n`, `\t`, `\r` escaped, every character Python does not print as itself (controls, U+2028, a non-breaking space) as `\u{...}`; a lone surrogate is refused |
| a string in a variable, a parameter, a return (several too), a tuple of returned values, a record field (a dataclass, a named tuple, a class, a dict with constant keys), a default (`def f(sep=", ")`, `name: str = "x"`) | `Str` | `def f(s: str)` and `list[str]` annotations are read as the others are |
| `s + t`, `s * n`, `n * s`, `s += t` | the same | `n` an int (`n <= 0` gives `""`); a float, a Bool or a string times a list is refused (TypeError, or not translated) |
| `==`, `!=`, `<`, `<=`, `>`, `>=` of strings | the same | code point order, as Python; `==` of a string and a number, a Bool or None is `false` (`!=` is `true`); an order between them is refused (TypeError) |
| `sub in s`, `s in lst`, `x in ("a", "b")`, `not in` | the same | |
| `len(s)`, `s[i]`, `s[a:b:k]`, `s[-1]`, `s[::-1]` | the same | Python's positions: negative from the end, slice bounds clamped, any step but 0; an index out of range stops with `IndexError: string index out of range`; a float or Bool index is refused |
| `if s:`, `not s`, `while lst:` | `len(s) > 0` | |
| `str(x)`, `repr(x)`, `format(x, spec)`, `f"..."`, `"...".format(...)`, `"..." % (...)` | the text of each value as Python writes it | an int by `fmt(n, "d")` (Viv's `str(3)` is `"3.0"`), a float by `str` (its repr), a Bool by `py_bstr` (`True`, `False`), a string as itself or by `repr` (Python's quotes and escapes), a list of floats, ints or strings as Python prints it; a NumPy array, a complex number, a list mixing ints and floats and an object are refused |
| format specs | `fmt(x, spec)` | numbers: `[[fill]align][width][.precision]` with type `d`, `e`, `f`, `g` (an int's digits with no type); strings: `[[fill]align][width][.precision]`; a sign, `,`, `%`, `#`, `0`, `=`, upper case and the int bases are refused |
| `"%s|%5.2f|%-4d|%r|%%" % (a, x, n, s)` | concatenated text | `%s %r %d %i %f %e %g %%` with a width, a precision and `-`; `%d` of a float truncates as Python does; a mapping key, `*` and other flags or types are refused |
| `"{} {:>6.2f} {name!r}".format(a, x, name=s)` | concatenated text | a literal template; each argument is evaluated once, in order; one that calls a def must appear once, in its place |
| `sep.join(x)` | `join(sep, x)` | `x` a list of strings, a string (its characters), a generator or a comprehension of strings (built as a list first), `map(str, xs)` |
| `s.split()`, `s.strip()`, `s.upper()`, ... | `split(s)`, `strip(s)`, `upper(s)`, ... | the rules; `split(None, n)`, `splitlines(True)`, `startswith` with a tuple or a start, and `casefold`, `title`, `capitalize`, `swapcase`, `partition`, `translate`, `expandtabs`, `encode` are refused |
| `float(s)`, `int(s)`, `int(s, base)`, `chr(n)`, `ord(c)` | the same | CPython's parsing and messages; an `int` past 2^53 stops (`py_int`) |
| `["a", "b"]`, `[]` grown by `lst.append(s)`, `[f(x) for x in ...]` of strings, `s.split(...)` | a list of strings | immutable in Viv: a list py2viv owns (the proof of "Functions and lists") grows by `lst = append(lst, s)`, and `pop`, `insert`, `remove`, `extend`, `clear`, `reverse` and `sort` rebind it to a new list; `lst[i] = s` and `lst += [...]` are refused (write `lst = lst[:i] + [s] + lst[i + 1:]`, `lst = lst + [...]`) |
| `lst.index(s)`, `lst.count(s)`, `sorted(lst)`, `sorted(lst, reverse=True)`, `sorted(s)`, `list(s)`, `lst == other`, `lst + other`, `lst[a:b]` | the same, or `py_chars(s)` | a missing item stops with `ValueError: 'x' is not in list`; `sorted(..., key=...)` of strings is refused |
| `sorted(set(s))`, `sorted(set(lst))`, `sorted(set(s), reverse=True)`, `len(set(s))` | `unique(s)`, `unique(lst)`, `unique(s)[::-1]`, `len(unique(s))` | Viv has no set: these are the uses of a set of strings with one answer; any other `set(...)`, and a set of numbers, is refused |
| `for c in s`, `for w in lst`, `enumerate`, `zip`, `reversed`, comprehensions over them | a loop over the index | |
| `print(s)`, `print(a, s, sep=...)`, `print(*words)` | `print(s)`, `print(fmt(a), s)`, `print(join(sep, words))` | a starred list of strings alone in the print; any other starred argument is refused |
| `raise E(f"...")`, `assert c, msg` with a computed message | `stop(py_excline("E", text))` | the message is Python's exactly (the type alone for an empty one) |
| `open(p).read()`, `Path(p).read_text()` | `read_text(p)` | UTF-8, `\r\n` and `\r` read as `\n` (Python's universal newlines) |
| `open(p).readlines()`, `for line in open(p)`, `with open(p) as f:` then `f.read()`, `f.readlines()` or `for line in f` (once) | `py_keeplines(read_text(p))` | each line with its `\n`, the last without one when the text does not end with it |
| `with open(p, "w") as f:` then `f.write(s)`, `open(p, "w").write(s)`, `Path(p).write_text(s)` | the text gathered, then `write_text(p, text)` | written when the block ends (a Python program that raises inside the block leaves part of it written; py2viv's leaves none); mode `"a"`, binary files, other encodings, `newline=`, a file read twice in a block and `input()` are refused |

A Python whose Unicode data is not CPython 3.11's (3.12 has Unicode 15.0) could answer `upper`,
`isalpha` or `repr` for a character added since differently from Viv's tables: under one, a
string literal beyond ASCII is refused, and text read at run time (a file, `chr(n)` of a computed
`n`) is checked to be ASCII (`py_ascii_text` stops otherwise).

`diff.py` reads a string value back from a printed line by the text Python printed (recorded
with its arguments), compares every line of text byte for byte (a line that differs from
Python's only around equal numbers was accepted before: RULES.md, Fixed holes, diff-text), reads
the output with no newline translation, runs the interpreter and the native program in the
directory Python ran the program in, and attaches to the wasm run each file the program names
that exists (a browser reads only the files attached). A Viv string builtin that stops stops with
CPython's message; diff.py compares it with Python's exception line (`ValueError: ` before
`could not convert string to float: 'x'`, the table `VIV_ERRORS`). A file that does not exist
stops the three engines with different words (the browser's names the attached files): such a
run fails.

The generated tests (`check_rules.py`): each string rule's `tests` (a Python string is written
`{ str = "..." }`) at every kind of text (empty, one character, accented, non-BMP, combining
marks, every kind of space, separators, case mappings that change the length, control
characters, quotes and backslashes, digits, line breaks), a number and a list in the string's
place (TypeError), each argument by keyword, one too many and missing; and the pseudo-rules
`strings.operators`, `strings.subscripts`, `strings.text` (str, repr, specs, %-formatting and
str.format of -0.0, 0.1 + 0.2, 1e16, 1e-7, 5e-324, NaN, the infinities, 2**52, ints and Bools),
`strings.lists`, `strings.print` and `strings.files`, which require CPython's text byte for byte.
`tests/test_py_str.py` runs longer programs in the three engines against CPython, the stops and
the refusals.

## Refused, with what to write instead

| Python | refusal hint |
|---|---|
| f-string specs outside fmt's table (`+.3f`, `,`, `.1%`, `.3`, `010.2f`, `=10`, upper case), `end=` | the table in "How printing works" |
| a slice with a variable step, a negative step of a matrix (`A[::-1]`), a subscript of a slice (`a[1:3][0]`), `...` in a subscript, a store through `None` | write the loop over the indices; one subscript per axis |
| a mask or an index array beside a range (`A[1:, idx]`), a mask beside another index (`A[m, 0]`), an index array of floats, an index array into a Python list, a mask of another rank than the array or a vector | take the range first, or turn the mask into indices with `np.nonzero(m)[0]` |
| `/` of two arrays one of which is complex, comparisons of complex arrays, comparisons of Python lists | NumPy's meaning is entry by entry and Viv's is not; compare real and imaginary parts, or make the list an array |
| a slice, row, transpose or reshape of an array the scope also writes into (or writes through) | a NumPy view shares the array's memory, a Viv slice is a copy: `.copy()` in Python, or index the array |
| a call of a def that returns several values inside an expression, or as a value | unpack it on its own line: `a, b = f(x)` |
| a starred or nested unpacking target, returning a tuple from some paths and one value from others | unpack into names; return the same number of values from every `return` |
| a list method (`append`, `pop`, `insert`, ...) of a list another name, a caller, a container or a nested def may hold (a parameter, a list bound outside the scope), an empty list anywhere but a binding, `list + list` | build the list in the function that owns it, or allocate the final size with `np.zeros(n)` and fill by index |
| a row another name can reach put into a list (`L.append(r)`, `[r for r in M]`), in a program that writes into rows | put a copy in: `list(r)`, `r.copy()` |
| a generator expression held in a variable or passed to a def, `tuple(...)`, `set(...)` other than `sorted(set(...))` or `len(set(...))` of strings | build a list with a comprehension; a set's iteration order (string hashing, which `PYTHONHASHSEED` randomizes) has no one answer |
| an `axis=` that is not an int literal, a tuple of axes, `dtype=`, `out=`, `ddof=` | R9: write the loop |
| a sum, mean, variance or product of a matrix NumPy may hold in a transpose's memory order (other than `A.T` itself) | NumPy sums in memory order; reduce the untransposed array or `.copy()` it |
| `np.append`, `np.tile`, `np.repeat`, `a.squeeze()`, a join of arrays of rank 3 or more, `np.meshgrid` of more than two vectors or not unpacked into two names, a reshape of a complex array or with `order='F'` | R10/R17: allocate the new shape and copy |
| inheritance, class decorators other than @dataclass, `__new__`/`__getattr__`/`__setattr__`/`__del__` and the other hooks, operators on objects, `==`/`is`/`isinstance`, attributes created outside `__init__` or set only on some paths, `self` used as a value in `__init__`, None-valued attributes, linked structures, lists of objects | "Classes, dataclasses, named tuples and dicts" above |
| an object whose fields are rebound, given a second name (`q = p`, a field, a list, `return p` of a parameter), passed twice to a def that changes it, changed inside a def while it is a module variable, or changed by a call inside an expression | Python shares one object, Viv copies a record: use one name, and call the method on its own line |
| a dict with string keys computed at run time or string values, a dict whose values are lists, arrays, Bools or other dicts, a Bool key, `d.get(k)` or `d.setdefault(k)` without a default, `d.items()` outside a loop, `popitem`, `fromkeys`, `most_common`, dict comparisons, printing a Counter or a defaultdict, a loop over a dict that may both add and remove keys, `set` | a Viv map holds numbers under number keys (string keys are being built on another branch), and a record the keys of its literal (whose values may be strings); key an array by an integer index (a set of numbers for membership is not built: its iteration order is CPython's hash order) |
| the string operations "Strings" lists as refused (tuples of strings, a store into a string or a list of strings, `lst += [...]`, `casefold` and the other methods Viv leaves out, `%x` and the other printf types, `str(A)` of a NumPy array, `sorted(words, key=...)`, `input()`, `open` other than read whole or written in a `with`) | the hint at each |
| `try`/`except`, a bare `raise`, `raise` inside a test, an exception with several arguments | nothing can catch a Viv `stop`; pytest would record an error in a test and move on, where Viv stops |
| enum members other than as listed in "Ints, enums and optionals" (`.value`, `auto()`, `C(1)`, iteration, arithmetic, members in containers), IntEnum and Flag; an Optional used other than through `if x is None` / `if x is not None` | "Ints, enums and optionals" above |
| `with`, `yield`, `async`, `match`, decorators, `nonlocal`, `global` rebinding, a def nested two levels deep, a lambda returned or stored | named per construct |
| a variable holding two types inside an `if` or a loop; printing a number that is an int on some paths and a float on others, or computing with it where Python's int and float results differ (`*`, `+`, `-`, `//`, `%`, `**` beside an int, unary `-`) | use two names; convert with `float(...)` |
| printing a tuple, a list of tuples, or a list of ints and floats | Python writes tuples with parentheses and each number as it is; print the numbers |
| a function called with a number in one place and an array in another | write two functions |
| `import` of anything but `math`, `numpy`, `viv` (and a use of a refused import) | the module name and "not in the numeric subset" |
| `np.random.*` without a seed, the legacy normals, other distributions | "np.random: NumPy's streams, draw for draw" above |
| integer literals above 2^53, except where an Int is bound or computed ("Ints, enums and optionals") | Viv numbers are float64 |
| bit operations on floats, bools or arrays | Python's `&` of two bools is a bool; of floats a TypeError |
| `a or b`, `a and b` of numbers used as a value | Python returns one of the operands |
| a row of a matrix the code also writes into (`r = A[0]; A[0, 1] = 5`) | a NumPy row is a view, a Viv row a copy: index the matrix |
| a float stored into a NumPy integer array, `np.array(..., dtype=int)` of floats | NumPy truncates; make the array float |
| `np.linalg.eigh` | each eigenvector's sign (and a degenerate pair's basis) is the solver's choice; use `np.linalg.eigvalsh` |
| `np.std`/`np.var` inside a def `grad` differentiates | write it out with `sum(...) / n` |
| `round(x, n)`, `np.round` of an array | print with a format, or scale, round, unscale |
| `np.linalg.det`, `eig`, `svd`, `qr`, `lstsq`, `pinv`, `cond` | not mapped yet (`eig_general`, `lib/svd` exist) |
| `math.factorial`, `comb`, `gcd`, `fsum`, `ulp`, `log1p`, `expm1` | not mapped |

## What is exact

Every difference between the Python and the translation is in one of three
places: exact (a), a stated and checked tolerance the program must declare (b),
or refused (c), at translation or, where only the value can tell, at run time
with a located `stop`. Nothing is left silent.

| difference | how it is handled |
|---|---|
| summation order | (a) `np.sum`, `mean`, `var`, `std`, `trace`: NumPy's order, bit for bit: from 0.0, blocks of 8192 entries each summed pairwise; along the last axis the same per entry, along any other axis the slabs added in order; a slice as its copy; a transpose as its original (`probe_numpy_order.py`, results/axis_order.txt); the builtin `sum`: left to right over an array, CPython 3.12's compensated sum over a list of floats; `cumsum`, `prod` left to right, as NumPy; (b) inside a def `grad` differentiates, `np.sum`/`np.mean` run left to right (Viv's `grad` cannot follow the pairwise recursion), listed as inexact |
| libm (`exp`, `log`, `sin`, `pow`, `**` of floats, `atan2`, `erf`, `gamma`, ...) | (b) Viv's own libm (lang/viv/src/vmath), identical in its three engines, differs from the platform's in the last bit on some inputs (`examples/mapping.py`: all within 5.3e-15 absolute, 4.6e-16 relative). `sqrt`, `abs`, `floor`, `ceil` are correctly rounded in both: (a) |
| `log10`, `log2`, `log(x, b)`, `hypot` | (b) written as `log(x) / log(b)` and `sqrt(x*x + y*y)` (`log10(1000)` is 2.9999999999999996) |
| BLAS: `@`, `np.dot`, `.dot`, `np.linalg.norm` | (b) NumPy's BLAS sums in its own blocked order |
| LAPACK and pocketfft: `solve`, `inv`, `cholesky`, `eigvalsh`, `fft`, `ifft` | (b) Viv's LU, Cholesky, Jacobi and FFT are its own |
| eigenvector signs | (c) `np.linalg.eigh` is refused |
| complex division, `abs` of a complex number, complex `sqrt`/`exp`/... | (b) CPython's and NumPy's formulas are their own |
| `grad`, `value_and_grad` | (b) Viv's reverse mode against the reference's complex step (pyshim/viv.py) |
| broadcasting, masks and integer indices | (a) the operands are expanded by copying entries (`take`), then combined by one IEEE operation per entry, as NumPy's loop; no multiply-add is fused (the C backend compiles with FP_CONTRACT OFF, wasm has no fused instruction, Rust does not contract); gathers and scatters copy entries in row-major order; where NumPy raises, the run stops with the start of NumPy's line (the rest names shapes or an index, which diff.py compares as a prefix) |
| `-0.0` | (a) `fmt` prints `-0.0`; a zero int prints `0` (`py_int`, `+ 0.0`); `%`, `//`, `np.full`, `np.sign`, `np.round`, `max`/`min` keep CPython's and NumPy's signs of zero; diff.py compares signs of zero |
| integers above 2^53 | (c) literals refused; a computed int stops (`py_int`) where a double would round; `**` of ints is exact by squaring; (a) an int annotated `int` is a Viv Int, exact to 2^63, and (c) past 64 bits its arithmetic stops at run time |
| int or float | (a) py2viv follows Python's static type for printing; (c) a value that is an int on one path and a float on another cannot be printed |
| exceptions CPython raises | (a) ZeroDivisionError, math's ValueError/OverflowError, `int`/`round` of NaN or infinity, a negative shift: the translation stops with the same line; (c) a NumPy scalar (not a Python float) divided by zero gives inf in NumPy and stops here |
| index errors | (c) a negative index computed at run time counts from the end in Python and stops Viv (out of bounds), except beside a range (`A[:, j]`), where `py_ix` counts it from the end; (a) an integer index array counts negative entries from the end and stops with NumPy's IndexError at one out of bounds |
| `sorted`/`np.sort` of NaN, or of 0.0 with -0.0 | (c) stops at run time |
| strings | (a) Viv's strings are Python's by construction (code points, CPython 3.11's Unicode tables, Python's split, strip, find, repr, `float` and `int` parsing); a string builtin that stops stops with CPython's message, which diff.py reads as the exception's line; (c) under a Python whose Unicode data is not 3.11's, a literal beyond ASCII is refused and text read at run time must be ASCII (`py_ascii_text` stops otherwise); an `int(s)` past 2^53 stops (`py_int`) |
| an exception inside a test | (c) pytest records an error and moves on; Viv stops the run (diff.py reports the ends differ) |

`diff.py` compares every number bit for bit (and -0.0 against 0.0) unless the
program states a tolerance in a comment, `# py2viv: rtol=1e-12 atol=0`, or on
the command line. Its report lists the inexact operations the translation used
(from the manifest), so the reason for a tolerance is in front of whoever reads
it. The three Viv engines must agree byte for byte, and a run that stops must
stop with the same error line in all three.
