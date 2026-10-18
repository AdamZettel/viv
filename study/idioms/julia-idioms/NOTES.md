# Julia idioms in Viv

Source: https://github.com/TheAlgorithms/Julia @ `384424b762bbb40d43f31001beeca0c6ffda2f1f`
(shallow clone at `viv-wt/src/julia-idioms`), plus two programs written for this study in
the style of the Julia manual and the SciML tutorials (`ref.jl`, part B).

The question was which Julia idioms Viv has and which it lacks. Nine Viv programs cover
25 TheAlgorithms functions and two idiomatic-Julia examples. 37 tests, all passing in the
interpreter and in native C, and the two engines print byte-identical output.

## Files

| file | ports | Julia idioms the original uses |
|---|---|---|
| `odes.viv` | `runge_kutta_integration`, `euler_method`, `verlet_integration` | `push!`, tuple return, default argument `Δt = 1e-3`, tuple `tspan`, `2k2` |
| `quad.viv` | `riemann_integration`, `monte_carlo_integration`, `simpsons_integration`, `trapezoid_integration` | Symbol argument `:midpoint` with a default, float ranges `a:Δₓ:b`, `sum(f(i) for i in r)`, `sum(1:n) do i ... end`, Unicode names `Δₓ Σ xᵢ` |
| `linalg.viv` | `lu_decompose`, `determinant`, `gauss_jordan`, `rotation_matrix`; Julia's `A \ b`, `A' * b` | tuple return `L, U`, `prod([... for i in 1:n])`, `@. A[i, :] = ...`, `A[:, 1:end-1]`, dispatch on `T<:Integer`, `@assert` with message, `θ` |
| `linreg.viv` | the whole `linear_regression.jl` script | `X'W .+ b - Y'`, `.^`, `1 / 2m`, tuple return of three values |
| `kmeans.viv` | `KMeans.cluster` and helpers | `Vector{Vector{T}}`, `push!` into a list of lists, `points[assign[i]]`, `new_clust != clusters` |
| `stats.viv` | `variance`, `pearson_correlation`, `median`, `mode`, `prefix_sum`, `DifferenceArray` | `.-`, `.*`, `.^`, `sort` (a copy), `Dict`, `empty!`, `append!`, `x += ` on an element |
| `numtheory.viv` | `eratosthenes`, `collatz_sequence`, `prime_factors`, `catalan`; three one-liners from the manual | `fill(true, n)`, `primes[p] \|\| continue`, `findall`, `append!`, `÷=`, `big(n)`/BigInt, generators with `if`, 2-D comprehension, `collect(0:0.25:2)` |
| `lorenz.viv` | SciML-style `lorenz!(du, u, p, t)` + fixed-step `rk4!` (ref.jl part B) | in-place `f!`, `Base.@kwdef struct` with defaults, `(; σ, ρ, β) = p`, keyword `dt =`, do-block callback, `@. u += ...`, `similar` |
| `dispatch.viv` | the manual's `Dual <: Number` + Newton; a `Shape` hierarchy (ref.jl part B) | struct, methods on `Base.:+` etc., `promote_rule`, abstract type, dispatch over a `Shape[]` vector, `sum(area, shapes)`, `argmax(f, xs)`, keyword `tol`, tuple return |
| `ref.jl` | the reference: `include`s the **unmodified** originals from the clone, writes `ref_values.viv` | |
| `ref_values.viv` | GENERATED: every reference number as a Viv global `REF_*` | |
| `ref.out`, `interp.out`, `native.out` | the reference's printout, and the two engines' output | |
| `probe/` | one small program per friction entry below | |

Run: `julia ref.jl` (2 s), then `viv run odes.viv` etc. from this directory.

**Randomness.** `riemann_integration`, `monte_carlo_integration` and the k-means points call
`rand()`. `ref.jl` loads each original into a module that defines its own `rand()` as Viv's
generator (Park-Miller MINSTD, `srand(s)` sets the state to `floor(|s| mod (2^31-2)) + 1`), so
both sides draw the same numbers and no weights or data need exporting except the k-means
points, which Viv reads as `REF_KM_POINTS`.

## Sizes

The originals' own docstring examples where they have them, else small inputs:
RK4 on y' = y, h = 0.1 over [0, 1]; Euler and Verlet with 100 steps; quadratures with
n = 1000 (Riemann, Monte Carlo) and n = 100 (Simpson, trapezoid); 3x3 LU and Gauss-Jordan;
the linear-regression script as written (10 points, 1000 steps); k-means on 24 points, k = 3;
sieve to 100, Collatz from 27, Catalan 0..25 and 30; Lorenz with dt = 1e-3 for 2000 steps.
Nothing was shrunk from the originals. All nine programs together run in 0.07 s in the
interpreter.

## Agreement

Every check is an `assert_near`/`assert_allclose` against a Julia value in `ref_values.viv`.
The stated tolerances are the test tolerances; the measured miss was found by rewriting every
assert into a print of `|viv - julia|` and running it.

| program | quantities | measured miss |
|---|---|---|
| odes | 12 RK4 points (x and y), RK4 docstring case, Euler/Verlet at 4 points | 0 (bit-identical) |
| quad | Riemann midpoint/left/right/sin, Monte Carlo, Simpson, trapezoid | 0, except Riemann left and right 4.4e-16 and the builtin `simpson` 8.9e-16 |
| linalg | L, U, det, both Gauss-Jordan results, `A\b`, `A'b`, rotation | 0 |
| linreg | W, b, J[1], J[1000], 10 predictions | 0 |
| kmeans | converged centers, 2-iteration centers | 0 |
| stats | variance, Pearson, two medians, two modes, prefix sums, difference array | 0 |
| numtheory | 25 primes, Collatz length and peak, three factorizations, 27 Catalan numbers | 0 (exact integers) |
| lorenz | u(2), max z, x at t = 0.5/1/1.5/2, u(1) at rho = 14 | 0 |
| dispatch | f'(x) by deriv / complex step / hand duals, two Newton roots and iteration counts, total area, perimeter, argmax | 0 |

The two 4.4e-16 misses are one ulp in the sample points: Julia's float range
`a:Δₓ:(b-Δₓ)` computes its elements in extra precision (TwicePrecision), and the port computes
`a + k*dx` in plain doubles. The builtin `simpson` sums in a different order from the
original's.

The ports keep two behaviors of the originals that look like mistakes: RK4's
`while x < x_stop` with `x += h` overshoots to x = 1.0999999999999999 (12 points, not 11),
and `prime_factors` tests `number % i == 1` where it means `!= 0`, so it returns `[5, 19]`
for 97 and `[3, 3, 3]` for 35. Both are reproduced so the ports can be compared number for
number.

## Both engines

`tools/both-engines.sh` with the shared binary: all nine programs compile to C, build with
`cc -O2 -lm`, pass every test, and print output identical to the interpreter's
(`cmp interp.out native.out`). `dispatch.viv` compiled only after the change described in
F9. The direct wasm backend compiles all nine to `.wasm`, but running them needs
`www/public/viv/viv.wasm`, which is not built in this worktree, so the third engine was not
run.

## Line counts

Code lines only: no blanks, comments or docstrings. Viv counts leave out `test` blocks and the
`### ... ###` post header, and include each file's top-level driver lines. The Julia counts
for the TheAlgorithms files are the files; for lorenz and dispatch they are `ref.jl` part B.
Julia signatures spread over several lines (`runge_kutta_integration` takes 7) make the
Julia side a little longer than its content.

| program | Julia | Viv | where Viv's extra lines come from |
|---|---|---|---|
| odes | 49 | 60 | counting RK4's steps before allocating (F3), 2 x n result matrix (F4) |
| quad | 53 | 58 | range length (F13), explicit accumulation loops (F12) |
| linalg | 74 | 76 | `square_part` copy for `A[:, 1:end-1]` (F8), offset loops (F11) |
| linreg | 37 | 25 | shorter: the step is one line in both, and the Julia file has print/reshape lines |
| kmeans | 84 | 62 | shorter: a label array replaces the list of member lists |
| stats | 82 | 95 | `mode` without a Dict (F17), `first9` copy loop (F8) |
| numtheory | 40 | 100 | three length-unknown outputs (F3), gcd + recurrence for Catalan (F14), one-liners as loops (F12) |
| lorenz | 34 | 49 | params array + slot names (F2), `assign` loop for `.=` (F10), callback state as global arrays (F15) |
| dispatch | 38 | 86 | dual numbers as pairs with one def per operator (F2), shape tag dispatch (F2) |
| **total** | **491** | **611** | |

## Friction ledger

Severity: 1 cosmetic, 2 costs lines, 3 blocks or distorts the algorithm. Entries are ordered
by severity, then by how often they came up. Each has a probe in `probe/`.

### F1. `2x` is accepted and means `2`: juxtaposition silently drops the variable (syntax / tooling, severity 3, 6 sites in the references)

Julia multiplies a numeric literal written against a name: `2k2`, `4f(x)`, `2π`, `2m`,
`2(s.w + s.h)`. Viv lets two statements share a line with no separator
(`x = 3.0 y = 4.0` is two assignments, `probe/twostmts.viv`), so `y = 2x` parses as
`y = 2` followed by the expression statement `x`. No error in either engine; `cc` prints
"expression result unused" and the program prints the wrong number (`probe/juxtaposition.viv`):

```
k2 = 5.0
z = 1.0 + 2k2     # z: 3
```

- (a) had to write: `2.0 * k2` everywhere, and check by eye that no juxtaposition survived the port.
- (b) wished: either `2k2` means `2 * k2`, or the parser rejects a second statement on the same line / an expression statement with no effect.
- (c) Julia: `y += h * (k1 + 2k2 + 2k3 + k4) / 6` (runge_kutta_integration.jl).
- Found while porting; would have produced a wrong RK4 with passing syntax. Six places in the ported Julia use it (RK4 twice, Simpson `4f(...)`, linreg `1 / 2m`, dispatch `2π * s.r`, `2(s.w + s.h)`).

### F2. No user types that act like numbers: no methods on operators, no dispatch, records not in the interpreter (type system / missing data structure, severity 3, 3 sites)

The manual's `Dual <: Number` works because `+`, `*`, `sin` gain methods for `Dual` and every
generic function then runs on duals. Viv has no operator overloading and no dispatch, and a
`record` stops the interpreter (`probe/record_interp.viv`: "record types aren't supported
yet"). The dual-number port is a pair `[v, d]` with a def per operation, and the functions must
be rewritten in call form. The ugliest line of the study:

```
def h_dual(x) {
    return d_div(d_mul(d_exp(d_sub(d_const(0.0), x)), d_sin(d_mul(d_const(3.0), x))), d_add(d_const(1.0), d_mul(x, x)))
}
```

- (b) wished: `record Dual { v: Num, d: Num }` plus `def +(a: Dual, b: Dual)`, so that `h(Dual(0.7, 1.0))` runs the existing `h`.
- (c) Julia: `Base.:*(a::Dual, b::Dual) = Dual(a.v * b.v, a.d * b.v + a.v * b.d)`; `derivative(f, x) = f(Dual(x, 1.0)).d`.
- The same gap turned the `Shape` hierarchy into rows `[kind, a, b, c]` with `if s[0] == CIRCLE` chains (no vector of mixed structs), and `LorenzParams(ρ = 14.0)` into an array with `SIGMA = 0, RHO = 1, BETA = 2` slot names.
- Two substitutes worked and are the idiomatic Viv answer (see Wins): `deriv` on an inline lambda, and the complex-step derivative through the generic def.

### F3. No growable array: `push!`, `append!`, `empty!` (missing data structure, severity 2, 7 sites)

Every function whose output length is not known in advance needed a count-first pass or a
fixed buffer plus a trim. RK4 is the worst: to reproduce Julia's 12 points it replays the
solver's own float loop to count them:

```
def rk4_points(x0, h, x_stop) {
    x = x0
    n = 1.0
    while x < x_stop {
        x += h
        n += 1.0
    }
    return n
}
```

- (b) wished: `out = []` then `push(out, x)`.
- (c) Julia: `push!(output_x, x)`.
- Sites: RK4 (count pass), `collatz_sequence` (count pass), `eratosthenes` (count while sieving, then fill), `prime_factors` (buffer of 64 + `first_n`), `mode` (buffer + trim, and `empty!(result)` became `found = 0`), k-means member lists (replaced by labels), `samples` in the Lorenz callback (sized 4 by hand).

### F4. One return value, no tuple destructuring (type system, severity 2, 6 sites)

`L, U = lu_decompose(A)`, `W, b, J_history = linear_regression(...)`, `xs, ys = rk4(...)`,
`r, its = newton(...)`, `s, e = tspan`. Viv: `x, y = 1.0, 2.0` is a parse error
(`probe/tuple_destructure.viv`). Workarounds used: out-parameters (`lu_decompose(A, L, U)`),
a scalar out-parameter as a one-element array (`b_out[0] = b` in linreg), two series as the
rows of one matrix then `slab(rk, 0)` (odes), `[root, iterations]` packed in one array
(Newton), and `tspan` split into two positional arguments.

- (b) wished: `return L, U` and `L, U = lu_decompose(A)`.

### F5. No default or keyword arguments (syntax, severity 2, 8 sites)

`approx::Symbol = :midpoint`, `h = 1.0e-2`, `Δt = 1.0e-3`, `rk4!(...; dt = 0.01)`,
`newton(f, x; tol = 1e-14, maxiter = 50)`, `Base.@kwdef` defaults, `LorenzParams(ρ = 14.0)`.
`def g(x, h = 0.1)` and `f(1.0, dt = 0.1)` are both parse errors (`probe/default_arg.viv`,
`probe/kwarg.viv`). Every call site passes every argument, positionally:
`rk4(ignore_step, lorenz, u2, 0.0, 1.0, lorenz_params(10.0, 14.0, 8.0 / 3.0), 1.0e-3)`.

### F6. `deriv` sees only an inline lambda body; through a def it errors, through a parameter the C backend refuses (missing builtin / engine gap, severity 2, 3 sites)

`deriv(fn(x) -> g(x), 2.0)` stops with "`deriv` cannot differentiate through `g`"
(`probe/deriv_through_def.viv`), and a bare def name is refused. `deriv(f, x)` with `f` a
parameter runs in the interpreter and is a compile error in C (`probe/deriv_param.viv`).
So a Newton solver generic in `f` cannot use `deriv`; the two formulas were typed twice
(once as defs, once inline in the `deriv` test), and Newton uses the complex step instead.

- (b) wished: `deriv(g, x)` for any def built from differentiable operations, and `deriv(f, x)` inside a higher-order def in both engines.
- (c) Julia: `derivative(g, 2.0)` works for any generic `g`, which is the whole point of the Dual idiom.

### F7. No Symbols or enums (missing data structure, severity 2, 2 sites)

`:midpoint / :left / :right` became `MIDPOINT = 0`, `LEFT = 1`, `RIGHT = 2`, and shape kinds
`CIRCLE = 0` etc. The original's `else throw(ArgumentError("... :$approx ..."))` branch has
no counterpart: a wrong code falls into the midpoint default. Strings are not values
(`probe/error_call.viv`), so there is nothing to name a case with.

### F8. No slices or views (missing builtin, severity 2, 4 sites)

`A[:, 1:(end-1)]` needed a `square_part` def with two loops; `data[1:9]` a copy loop;
`a[1:3]` is a parse error (`probe/slice.viv`). Rows are covered: `slab(A, i)` and
`setslab(A, i, S)` read and write `A[i, :]`, so Gauss-Jordan's row operations stayed one line
each. Columns are not.

### F9. The C backend gives each def one argument type; the interpreter accepts two (engine gap, severity 2, 1 site)

`newton(g, ...)` called `g(x)` with a real and `complex_step` called `g(complex(x, eps))`.
The interpreter runs it; C stops with "argument 1 of `g` should be a complex number but is a
number". The port evaluates f at a complex point both times:
`dx = re(f(complex(x, 0.0))) / complex_step(f, x)`. Julia compiles a method instance per
argument type.

### F10. No in-place broadcast assignment; `a += ...` on an array parameter silently rebinds (syntax / type system, severity 2, 2 sites)

`@. u += dt / 6 * (k1 + 2k2 + 2k3 + k4)` updates the caller's `u`. In Viv, `u .= ...` is a
parse error (`probe/dot_assign.viv`), and `a += 1.0` or `a = a + 1.0` inside a def compiles,
runs and leaves the caller's array unchanged (`probe/rebind.viv`). The port writes a
four-line `assign(a, b)` loop and calls `assign(u, u + dt / 6.0 * (...))`. `@. tmp = u + dt/2*k1`
became a fresh array per stage, which costs allocations but no lines.

### F11. `for` has no start, step, or collection (control flow, severity 2, 8 sites)

`for j in i:n`, `for c in 2:k`, `for i in 2:div(n, p)`, `for x in 1:100`, `for p in points`.
Viv has `for i to n` (0 .. n-1) only (`probe/for_in.viv`), so each became
`for jj to n - i { j = i + jj }` or index arithmetic like `is_prime[p * (i + 2.0)]`.

### F12. No generators, comprehensions or do-blocks (syntax, severity 2, 7 sites)

`sum(f(i) for i in r)`, `sum(1:(n/2)) do i ... end`, `prod([L[i, i] for i in 1:n])`,
`[i * j for i in 1:3, j in 1:4]`, `sum(x^2 for x in 1:100 if isodd(x))`, `sum(area, shapes)`,
`argmax(area, shapes)`. Each is an accumulator and a loop in Viv (the odd-squares one-liner is
seven lines). Lambdas are single expressions, so a do-block body with statements has to be
a named def.

### F13. No float ranges (missing builtin, severity 2, 2 sites)

`(a+Δₓ/2):Δₓ:(b-Δₓ/2)` and `length(0.1:0.1:0.9)`: the port computes the count as
`round((hi - lo) / dx) + 1` and the elements as `lo + k * dx`, which differs from Julia's
range by one ulp at some points (measured 4.4e-16 in the left and right Riemann sums).
`linspace(0.0, 2.0, 9.0)` covers `collect(0:0.25:2)` exactly.

### F14. No integer type, no BigInt, no `^`, `÷`, `%` (type system, severity 3 at 1 site where the algorithm changed, severity 1 at ~10 others)

`catalan` computes `factorial(2n) ÷ (factorial(n)^2 * (n + 1))` in BigInt. In doubles,
`factorial(2n)` passes 2^53 at n = 10, past which a double no longer holds every integer, so
the port uses a different algorithm: the
recurrence C(k+1) = C(k) 2(2k+1)/(k+2) with a gcd taken out first so no intermediate passes
2^53, which gives C(30) = 3814986502092304 exactly. C(31) = 14544636039226909 is past 2^53, so from there on no Viv
number can hold the answer. Elsewhere `x^3` became `x * x * x` (`^` is "unexpected character",
`probe/caret_pow.viv`), `÷=` became `floor(number / i)`, `%` became `mod`.

### F15. Closures cannot update what they capture (type system, severity 2, 1 site)

The Lorenz do-block keeps a running max and appends samples. Viv lambdas are expressions,
and a def cannot rebind a global scalar (`n = n + 1.0` inside a def is refused), so the
callback is a def writing global one-element arrays:

```
zmax = [-1.0e300]
taken = zeros(1)
def record_step(t, u) {
    zmax[0] = max(zmax[0], u[2])
    ...
        taken[0] = taken[0] + 1.0
```

Julia needed `Ref(-Inf)` for the max too; the difference is the second def and the counter.

### F16. No `'` or `\`; no vector dot product (syntax, severity 1, 9 sites)

`A'` and `A \ b` are "unexpected character" (`probe/adjoint.viv`, `probe/backslash.viv`).
`transpose(A) @ b` and `lin_solve(A, b, x)` (an out-parameter) do the work. Linreg's
`X'W .+ b - Y'` became `transpose(X) @ W + b - transpose(Y)`, three times in three lines.
`b @ b` is an error, so a dot product is `sum(a * b)`.

### F17. No Dict (missing data structure, severity 2, 1 site)

`mode` counts with `Dict()`; the port keeps parallel `keys`/`counts` arrays with a linear
search, 12 lines where Julia has 5.

### F18. No `throw` / `error` with a message (control flow, severity 2, 4 sites)

`throw(DomainError(...))` in `determinant`, `runge_kutta_integration` and `catalan`, and
`@assert det ≠ 0 "Must insert a non-singular matrix"` in `gauss_jordan`. Viv has
`assert_true`, which records a failure and keeps going; there is no way to stop with a
message. Separately, the interpreter accepts a call to the unknown function `error` in a
branch that is not taken (`probe/error_unchecked_branch.viv` prints `2`); the C backend
rejects it.

### F19. Unicode identifiers crash the lexer (tooling, severity 2, ~12 names)

`Δx = 0.5` panics: "byte index 1 is not a char boundary; it is inside 'Δ'"
(`probe/unicode_ident.viv`). A Rust panic, not a diagnostic. Julia code names things
`Δₓ`, `Σ`, `xᵢ`, `θ`, `σ ρ β`; each was renamed to ASCII (`dx`, `total`, `theta`,
`SIGMA`). Unicode in comments and test names works.

### F20. Array `==` is not elementwise or whole-array (missing builtin, severity 1, 1 site)

`new_clust != clusters` in k-means. `[1.0, 2.0] == [1.0, 2.0]` stops the interpreter with
"only + - * / are supported elementwise" (`probe/array_eq.viv`). Written as
`max(abs(A - B)) == 0.0`.

### F21. No `copy`, no `length`/`size` of all dimensions (missing builtin, severity 1, 7 sites)

`sorted = nums + 0.0` is the copy before an in-place `sort` (sort writes through any alias,
`probe/sort_aliases.viv`). `length(Y)` of a 1 x m matrix is `len(transpose(Y))`;
`size(A, 1)` does not exist (`probe/size.viv`).

### F22. No `+=` on an array element (syntax, severity 1, 3 sites)

`diff_arr[l] += x`, `dict[i] += 1`: "compound assignment into an array element is not
supported" (`probe/elem_pluseq.viv`), written `a[i] = a[i] + x`. The message says what to
write.

### F23. No `if` expression / ternary (syntax, severity 1, 2 sites)

`n = n % 2 == 1 ? 3n + 1 : div(n, 2)` and `return if len % 2 == 0 ... else ... end`.
`y = if c { a } else { b }` is a parse error (`probe/if_expr.viv`); both became
statement `if`s.

### F24. A test that errors is first reported as PASS (tooling / error message, severity 1, hit once)

When a test's first assertion hit the `deriv` error, the output was
`PASS deriv gives ... (no assertions)` followed by the error. The exit code was 1, but the
PASS line is wrong.

### F25. Output: six significant digits; error lines point into the spliced source (strings/IO, tooling, severity 1, throughout)

Known from the other studies. All comparisons are asserts inside Viv against
`ref_values.viv`, so the six-digit `show` did not matter. With `#! use ref_values`, an error
in a 90-line file was reported at line 173.

## Wins

- **Arithmetic broadcasts without dots.** `d = a - mean(a)`, `sum(dx * dy)`,
  `u + dt / 2.0 * k1`: Julia's `.-`, `.*`, `.^` and `@.` are the default for `+ - * /` on
  arrays. `variance` and `pearson_correlation` are the same length as the Julia.
- **The linear-regression step reads like the Julia.** `W - alpha * (1.0 / m) * (X @ (transpose(X) @ W + b - transpose(Y)))` is Julia's line with `'` spelled `transpose` and `*` spelled `@`.
- **In-place `f!` works as in Julia.** An array parameter is the caller's array, so
  `lorenz(du, u, p, t)` writes `du` exactly like `lorenz!`, and `lu_decompose(A, L, U)`
  fills L and U.
- **Functions are values and lambdas close over variables.** `fn(x, y) -> y`,
  `runge_kutta_integration(fn(x, y) -> exp(x), ...)`, `newton(dottie, ...)`,
  `total(area, shapes)`: the same shape as Julia's anonymous functions and higher-order calls,
  in both engines.
- **One def, scalars or arrays** (in the interpreter): `def f(x) { return x * 2.0 }` takes a
  number or an array (`probe/win_generic_def.viv`), like an untyped Julia method.
- **Complex scalars through generic code.** `complex_step(g, 2.0)` runs the unchanged
  `g` on a complex argument and gives g'(2) = 10 exactly and h'(0.7) to the last bit; this is
  the closest Viv gets to Julia's duck-typed numeric code.
- **`deriv` is exact forward-mode autodiff** on an inline expression, in both engines, with no
  Dual type to write.
- **Row operations.** `setslab(A, j, slab(A, j) - slab(A, i) * A[j, i])` is Julia's
  `@. A[j, :] = A[j, :] - A[i, :] * A[j, i]`.
- **Builtins that replace a function:** `simpson`, `trapezoid` (agree with the originals to
  8.9e-16 and 0), `lin_solve` for `\`, `linspace` for a float range, `sort`, `max(abs(A - B))`.
- **The error messages coach Julia/Python habits:** `for x in a` names `for i to n`;
  `a[i] += x` names `a[i] = a[i] + x`; reading a global you later assign explains the
  hoisting rule.
- **Agreement:** 37 checks, all bit-identical to Julia except three at 1-2 ulp, and the
  interpreter and native C print the same bytes.

## Julia idioms: has / lacks

| Julia idiom | Viv |
|---|---|
| dot broadcasting `.+ .- .* ./` | has, implicitly, for `+ - * /` and many math builtins; no `.==`, no `.^` (`pow(a, s)`), no row broadcast |
| fused in-place `@.` / `.=` | lacks (F10) |
| comprehensions, generators | lacks (F12) |
| do-blocks | lacks; a named def instead (F12, F15) |
| multiple dispatch | lacks (F2); tag + `if` |
| structs | C backend only; not in the interpreter (F2) |
| operator overloading on user types | lacks (F2) |
| keyword / default arguments | lacks (F5) |
| tuples and destructuring | lacks (F4) |
| ranges `a:s:b`, `i:n` loops | lacks; `for i to n`, `linspace` (F11, F13) |
| views and slices | rows only via `slab`/`setslab` (F8) |
| mutating `f!(du, u)` | has |
| closures | has, read-only captures (F15) |
| Unicode names | crashes the lexer (F19) |
| `\`, `'` | `lin_solve`, `transpose` (F16) |
| `2x` juxtaposition | silently wrong (F1) |
| Symbols, Dict, push!, BigInt, Int | lack (F7, F17, F3, F14) |
| `@assert`, `throw` | `assert_true` only (F18) |
