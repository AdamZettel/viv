// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The language-tour cells — like the SCF notebook, the concatenation must run
// at every prefix (cell N executes cells 1..N). Names are unique across cells.
export const cells = [
// ---- 1: numbers -------------------------------------------------------
`# A number in Viv is a 64-bit float unless it is declared an Int:
# 10 lexes as 10.0.
n = 10
show n
show n / 4.0

# print gives the bare value; show labels a variable with its name
# (n: 10 above) and prints any other expression bare:
print(sqrt(2.0))
show pow(2.0, 10.0)

# the math builtins: exp log sin cos tan sqrt pow abs floor ceil min max
show min(floor(2.7), ceil(1.2))`,
// ---- 2: control flow --------------------------------------------------
`# if / else, while, and \`for i to n\` — i runs 0 .. n-1.
total1 = 0.0
for i to 10 {
    total1 = total1 + i
}
show total1

# while, growing to the first power of two past 1000:
p2 = 1.0
while p2 < 1000.0 {
    p2 = p2 * 2.0
}
show p2

# break works the way you expect:
q = 0.0
while 1.0 < 2.0 {
    q = q + 3.0
    if q > 10.0 { break }
}
show q`,
// ---- 3: functions -----------------------------------------------------
`def fact(n) {
    "n! by recursion. Every number is a double, and a double holds integers exactly."
    if n <= 1.0 {
        return 1.0
    }
    return n * fact(n - 1.0)
}
show fact(10.0)
# exact to the last unit, not merely close:
show fact(20.0) - 2432902008176640000.0

# What a double cannot hold exactly is a fraction like 0.1. THAT is where
# equality fails, and where a tolerance belongs:
s = 0.1 + 0.2
show s - 0.3
assert_near(s, 0.3, 1.0e-12)

# type annotations are optional and checked at lowering:
def hyp(a: Num, b: Num) -> Num {
    return sqrt(a * a + b * b)
}
show hyp(3.0, 4.0)`,
// ---- 4: arrays --------------------------------------------------------
`# Array literals, elementwise arithmetic, and indexing.
v = [3.0, 4.0]
w = [10.0, 20.0]
show v + w
show v * w
show v * 2.0
show len(v)

# matrices are arrays of rank 2; zeros() allocates up to rank 4:
M = [[2.0, 0.0], [0.0, 5.0]]
show M[1, 1]
Z = zeros(2, 2)
Z[0, 1] = 7.0
show Z`,
// ---- 5: plotting convention -------------------------------------------
`# A figure is declared IN the program: figure(title, xlabel, ylabel), then
# plot(x, y, "label"). The interpreter emits both as records and whatever is
# hosting the program draws them -- adding nothing of its own.
n_t = 61
wave_x = linspace(0.0, 6.0, n_t)
wave = zeros(n_t)
for i to n_t {
    wave[i] = exp(-0.5 * wave_x[i]) * cos(3.0 * wave_x[i])
}
figure("A damped cosine", "t", "exp(-t/2) cos(3t)")
plot(wave_x, wave, "decay x oscillation")`,
// ---- 6: linear algebra ------------------------------------------------
`A = [[2.0, 1.0], [1.0, 3.0]]
show A @ transpose(A)

# eig_sym: Jacobi eigendecomposition of a symmetric matrix, ascending:
vals = zeros(2)
vecs = zeros(2, 2)
eig_sym(A, vals, vecs)
show vals

# the trace equals the sum of the eigenvalues — 2 + 3 = 5:
show vals[0] + vals[1]`,
// ---- 7: lambdas and higher-order builtins -----------------------------
`# fn(x) -> expr is a value; call it like anything else:
square_fn = fn(x) -> x * x
show square_fn(7.0)

# quadrature takes a lambda — integral of x^2 over [0,1], exact 1/3:
show simpson(fn(x) -> x * x, 0.0, 1.0, 40)

# deriv is exact forward-mode autodiff, not finite differences.
# d/dx [x sin(x)] at 2, against the product rule sin(2) + 2 cos(2):
show deriv(fn(x) -> x * sin(x), 2.0)
show sin(2.0) + 2.0 * cos(2.0)`,
// ---- 8: tests ---------------------------------------------------------
`# A test block runs its body and prints PASS/FAIL — programs grade themselves.
test "simpson integrates exp correctly" {
    assert_near(simpson(fn(x) -> exp(x), 0.0, 1.0, 60), exp(1.0) - 1.0, 1.0e-9)
}`,
// ---- 9: prose that compiles to nothing --------------------------------
`# narrate is the program narrating its own progress — a live channel,
# separate from data output, in every backend:
narrate "setting up"
answer = 6.0 * 7.0
narrate "answer computed"

#| Annotation markers stay ordinary comments to the compiler;
#| \`viv doc\` extracts them for code browsers.
show answer`,
// ---- 10: parallel for -------------------------------------------------
`# \`parallel for\` runs sequentially here in the interpreter;
# the C backend emits an OpenMP pragma for it.
psum = 0.0
parallel for i to 1000 {
    psum = psum + i
}
show psum`,
// ---- 11: records ---------------------------------------------------------
`# A record bundles named numbers and arrays into one value.
record Point { x: Num, y: Num }
record Polyline { xs: Arr, ys: Arr }

def norm2(p: Point) -> Num {
    return sqrt(p.x * p.x + p.y * p.y)
}

pt = Point { x: 3.0, y: 4.0 }
show norm2(pt)

# Copying a record copies its numbers; an array field is the same array.
line = Polyline { xs: zeros(3), ys: zeros(3) }
line2 = line
line2.xs[1] = 5.0
show line.xs

# A def can return several values; the caller unpacks them.
def extent(v: Arr) {
    return min(v), max(v)
}
lo, hi = extent(line.xs)
show hi - lo`,
// ---- 12: capstone — the variational minimum, in tour pieces ------------
`# Everything above, in one physics one-liner: the single-Gaussian
# hydrogen-atom energy from the SCF notebook, E(a) = 3a/2 - 2 sqrt(2a/pi).
pi_c = 3.141592653589793
best_a = 8.0 / (9.0 * pi_c)

def E_h(alpha) {
    "single-Gaussian variational energy for the hydrogen atom"
    return 1.5 * alpha - 2.0 * sqrt(2.0 * alpha / pi_c)
}
show E_h(best_a)

# at a true minimum the derivative vanishes — deriv checks it exactly:
show deriv(fn(a) -> 1.5 * a - 2.0 * sqrt(2.0 * a / pi_c), best_a)

test "the analytic optimum -4/(3 pi)" {
    assert_near(E_h(best_a), 0.0 - 4.0 / (3.0 * pi_c), 1.0e-12)
}`,
];
