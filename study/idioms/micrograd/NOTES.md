# micrograd in Viv

Source: https://github.com/karpathy/micrograd @ `7bc720e951fe422b8f8814aa5aa1b64121d26b4c`
(shallow clone at `viv-wt/src/micrograd`). Ported: `micrograd/engine.py` (the `Value`
engine), `micrograd/nn.py` (Neuron/Layer/MLP), the demo notebook's loss and training
cells, and both tests in `test/test_engine.py`. Not ported: the decision-boundary plot,
the minibatch branch of `loss()` (the demo calls it with `batch_size=None`), `__repr__`s.

## Files

| file | what |
|---|---|
| `micrograd.viv` | the port: tape engine, MLP, loss, training, tests |
| `ref.py` | runs the **unmodified** cloned micrograd on the demo; writes `moons.viv` |
| `moons.viv` | GENERATED: moons data, initial parameters, reference loss/acc/grad/final params, torch values for the engine tests |
| `interp.out`, `native.out` | the two engines' output (identical) |
| `ref.out` | the reference's per-step printout |
| `probe/` | the small programs behind the friction entries below |

Run: `python3 ref.py` (80 s), then `viv run ./micrograd.viv` (the `./` matters, see F10).

## Sizes

The demo's own sizes, not shrunk: `make_moons(n_samples=100, noise=0.1)`, seeds 1337,
`MLP(2, [16, 16, 1])` = 337 parameters, 100 SGD steps, `lr = 1.0 - 0.9*k/100`, SVM
max-margin loss + 1e-4 L2. One step's graph is 65,818 nodes.

| engine | time |
|---|---|
| CPython micrograd (reference) | 80 s |
| Viv interpreter | 21 s |
| Viv -> C, `cc -O2` | 0.10 s |

## Agreement

The random parts (data, `random.uniform` initial weights) come from the reference via
`moons.viv`. The port reproduces micrograd's graph op for op (subtraction as `a + (-1*b)`,
division as `a * b**-1`, `sum(..., start)` left to right), so only the order in which
gradient contributions are summed into a node differs (tape order vs micrograd's DFS order).

| quantity | measured miss | test tolerance |
|---|---|---|
| test_engine `sanity_check`: y and dy/dx vs torch | 0 (exact, like the original test) | 0 |
| test_engine `more_ops`: g, dg/da, dg/db vs torch | within 1e-12 relative | 1e-12 rel |
| step-0 gradient, all 337 parameters | 3.3e-16 to 4.4e-16 abs | 1e-12 |
| loss, every one of 100 steps | 3.5e-15 to 7.1e-15 relative | 1e-12 |
| accuracy, every step | exact | 0 |
| final 337 parameters | 6.7e-16 abs | 1e-10 |

The measured misses change from one `ref.py` run to the next: micrograd keeps parents in a
`set` of objects, whose order follows `id()`, so its DFS order, and the order it sums gradient
contributions, differs between Python runs. The two ranges above are two runs of `ref.py`.

Final: step 99 loss 0.0109784, accuracy 100%, in both.

**Both engines:** interpreter and native C print byte-identical output (`cmp interp.out
native.out`), all 6 tests pass in each.

## Line counts (code lines: no blanks, comments, docstrings)

| part | micrograd | Viv |
|---|---|---|
| engine (`engine.py` / tape + ops + backward) | 65 | 78 (+17 for the optional DFS topo comparison) |
| `nn.py` / MLP | 42 | 31 |
| loss cell | 17 | 29 |
| training cell | 9 | 21 (+25 of checks against the reference) |
| engine tests (micrograd half of test_engine) | ~30 | 43 (incl. topo check) |

## How the port is shaped

micrograd's `Value` is an object with a parent set and a `_backward` closure. Viv has no
objects, no statement-bodied closures, and records only in the C backend, so the graph is
a **tape**: six parallel arrays (`op`, `left`, `right`, `data`, `grad`, `expo`) and a
node is its index. Each op appends a node. Parents are always created before children,
so tape order is a topological order and `backward` is one reverse loop that dispatches on
`op`. The parameters are pushed first on every step, so parameter `i` is node `i`, and
`lib/params` gives each layer a row-per-neuron layout in one flat vector.

This is the same design as `lang/viv/examples/autodiff_reverse.viv`; it is what Viv
programmers already write, and it is how production tape autodiffs work. But it is a
different program from micrograd: the thing micrograd teaches (an object graph built by
operator overloading, walked by DFS) is replaced by an interpreter over opcodes.

## Friction ledger

**F1. No operator overloading; every expression is nested calls.** sev 3, syntax/type system, every expression (≈25 sites)
(a) had to write
```
c = add(c, add(add(value(1.0), c), neg(a)))
d = add(d, add(mul(value(3.0), d), relu(sub(b, a))))
g = add(g, div(value(10.0), f))
```
(b) wished: `c += 1 + c + (-a)`; `d += 3*d + (b - a).relu()`; `g += 10.0 / f`
(c) Python: `__add__/__radd__/__mul__...` on the class; Julia: `Base.:+(a::Value, b::Value)`.
This is the whole user-facing API of micrograd. `test "more ops"` is hard to check by eye against
`test_engine.py`.

**F2. No object graph: no objects, no references, no statement-bodied closures.** sev 3, missing data structure, the whole engine
(a) a tape of six parallel global arrays plus an `if op == ... else if` dispatch in `backward`
(see `backward` in micrograd.viv, 25 lines).
(b) wished:
```
record Value { data: Num, grad: Num, prev: [Value], back: fn() }
out.back = fn() { a.grad += out.grad; b.grad += out.grad }
```
(c) Python `out._backward = _backward` (engine.py); Julia `mutable struct Value; prev::Vector{Value}; back::Function; end`.
Lambdas are expression-only (`fn(i) { ... }` is `expected ->`, probe/p6.viv).

**F3. Records do not run in the interpreter, cannot go in arrays, cannot recurse, and take no operators.** sev 3, engine gap/type system, 1 (blocks the natural formulation)
(a) `record Value {...}` -> interpreter: `record types (Value) aren't supported yet` (probe/p1.viv);
C: `vs = [a, a]` -> `expected a number for a vector element` (p11); `a + a` -> `records do not
support operators` (p13). A `Tape { data: Arr, grad: Arr, n: Arr }` record bundling the columns
works in C (p12) but not in the interpreter, so the port uses six globals instead.
(b) `t = Tape {...}; add(t, a, b)` in both engines, or arrays of records.
(c) Python dataclass / Julia struct in a `Vector{Value}`.

**F4. Closures are first-class in the interpreter but not in C.** sev 2, engine gap, 1 (probe)
(a) avoided closures entirely.
(b) `def make_adder(k) { return fn(x) -> x + k }` — interpreter prints 4, C: `a closure is only
allowed as the first argument of simpson/trapezoid` (probe/p2.viv).
(c) Python/Julia closures.

**F5. A function cannot rebind a global scalar, so counters live in 1-element arrays.** sev 2, syntax (scoping), 2 counters / 8 sites
(a) `tape_len = zeros(1)` ... `i = tape_len[0]` ... `tape_len[0] = i + 1.0`; same for `topo_len`;
and `loss(acc: Arr)` returns accuracy through `acc[0]` because there are no tuples.
(b) `global n` / `n += 1` and `return total, hits / N`.
(c) Python `topo.append(v)`; `return total_loss, acc`. Julia `push!(topo, v)`; `return (l, acc)`.
The interpreter's error is clear (probe/p3.viv).

**F6. No growable arrays (`append`/`push!`).** sev 2, missing data structure, 2 (tape, topo list)
(a) `CAPACITY = 80000` sized by hand from a node-count estimate, plus manual length counters.
(b) `topo = []` ... `push(topo, v)`.
(c) `topo.append(v)` / `push!(topo, v)`.

**F7. A node id is a plain Num, so a constant where a node belongs is silently wrong.** sev 2, type system, ≈12 `value()` wraps
(a) every constant wrapped: `mul(score, value(-Y[s]))`, `value(1.0 / N)`, `value(1.0e-4)`.
`add(x, 2.0)` compiles and reads node 2 without complaint (probe/p10.viv: `addn(0, 2.0)`
returns `data[0] + data[2]`).
(b) a distinct `Node` type (or an `Arr(Int)` element type) so
`add(x, 2.0)` is a type error, or overloading that promotes `2.0` to a leaf.
(c) Python `other if isinstance(other, Value) else Value(other)`; Julia dispatch on `::Real`.

**F8. `show`/`print` are fixed at `%g` (6 significant digits).** sev 2, strings/IO, all output
(a) the demo prints `step {k} loss {loss.data}` at full precision; the port prints 6 digits and
proves agreement inside the program: embedded reference arrays, `assert_allclose`, and
`loss_miss = max(abs(loss_at - REF_LOSS) / REF_LOSS)`.
(b) `print(loss, digits: 17)` or a format string.
(c) `f"{x!r}"` / `@printf("%.17g", x)`.

**F9. No file input.** sev 2, strings/IO, 1
(a) `ref.py` writes the data and weights as Viv source (`moons.viv`, 29 KB of literals) and the
port reads it with `#! use moons`.
(b) `X = read_csv("moons.csv")`.
(c) `np.loadtxt` / `readdlm`.

**F10. `#! use` fails when the program is given without a directory.** sev 1, tooling, 1
(a) `viv run ./micrograd.viv`.
(b) `viv run micrograd.viv` from its own directory. It prints `no module by that name (looked
for lib/params.viv beside the program and under )`: the root is empty because
`module_dirs` canonicalizes the parent `""` (lang/viv/src/main.rs), which fails.
(c) n/a.

**F11. No comprehensions, `map`, or `sum` over a generator.** sev 2, syntax, 5
(a) `xs = zeros(2); xs[0] = value(X[s, 0]); xs[1] = value(X[s, 1])`; `data_loss = value(0.0);
for s to N { data_loss = add(data_loss, losses[s]) }`; same for the L2 term and neuron sums.
(b) `xs = [value(v) for v in X[s]]`; `reduce(add, losses, value(0.0))`.
(c) `list(map(Value, xrow))`, `sum(losses)`; Julia `Value.(xrow)`, `sum(losses)`.

**F12. No descending `for`.** sev 1, control flow, 1
(a) `i = out; while i >= 0 { ...; i = i - 1 }` in `backward`.
(b) `for i in reverse(0 to out)`. (c) `for v in reversed(topo)` / `for i in n:-1:1`.

**F13. No array copy builtin.** sev 1, missing builtin, 1
(a) `for i to NPARAM { grad0[i] = grad[i] }` (and `theta = P0` aliases rather than copies).
(b) `grad0 = copy(slice(grad, 0, NPARAM))`. (c) `grad[:n].copy()` / `grad[1:n]` (copies).

## Wins

- Byte-identical output in the interpreter and native C at the demo's full size; no engine could not run it.
- Speed: the interpreter (21 s) is 4x faster than CPython micrograd (80 s); native C is 0.10 s.
- `#! use lib/params`: each layer is a row-per-neuron matrix in one flat vector, so
  `parameters()`, `zero_grad()` and the Module class hierarchy disappear. The MLP is 31 lines vs 42,
  and the SGD update is one loop over `theta`.
- Parameter `i` is tape node `i`, so reading gradients back is `grad[i]` with no bookkeeping.
- Tape order is already topological: `backward` needs no DFS, no visited set, no recursion. The
  DFS (`build_topo`) still ports directly, recursion included, and a test checks it orders parents first.
- Tests as the unit of verification: `assert_allclose(loss_at, REF_LOSS, 1e-12)` checks all 100
  steps in one line; the sanity test matches torch exactly (tolerance 0), as the original asserts.
- Whole-array expressions: `max(abs(theta - REF_PFINAL))` in one line.
- `Bool` parameters (`nonlin: Bool`), `else if`, `or`, and indexing a call result (`layer(L3, h, false)[0]`) all read like Python.
