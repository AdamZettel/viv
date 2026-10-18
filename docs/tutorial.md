# A ten-minute tour of Viv

This tour runs from the command line. Build the compiler first
(`cargo build --release -p viv`, see the [README](../README.md)) and put
`target/release/viv` on your PATH, or try the same programs in the browser at
[viv-lang.org/workbench](https://viv-lang.org/workbench).

Every program below is a file in [`docs/tutorial/`](tutorial/), and
`docs/tutorial/check.sh` runs each one in the interpreter, as C and as wasm,
and requires the three outputs to be identical. The outputs shown are what
`viv run -q` prints (`-q` leaves out the call-graph records the browser uses).

## 1. A first program

[`01_first.viv`](tutorial/01_first.viv):

```viv
# The sum of the first n squares, three ways.
def sum_squares(n) {
    total = 0
    for i from 1 to n + 1 {
        total += i * i
    }
    return total
}

n = 10
print(f"sum of squares to {n:.0f}: {sum_squares(n):.0f}")
show sum([i * i for i from 1 to n + 1])
show n * (n + 1) * (2 * n + 1) / 6

test "the loop matches the closed form" {
    assert_true(sum_squares(n) == 385)
}
```

```text
$ viv run -q docs/tutorial/01_first.viv
sum of squares to 10: 385
385
385
PASS the loop matches the closed form
```

What to notice:

- `for i from 1 to n + 1` counts 1, 2, ..., n: the upper bound is excluded, as
  in Python's `range`. `for i to n` starts at 0.
- A number is a 64-bit float unless it is declared an `Int`, so `n / 2` never
  truncates. `{n:.0f}` in the f-string prints it without a decimal point.
- `print` prints its argument; `show` prints a value (with its name, for a
  variable), and is what most programs use to report results.
- A `test` block runs after the program and prints `PASS` or `FAIL` with its
  name. A failed test makes `viv run` exit with a nonzero status.

To compile the same program to C and run it:

```sh
viv docs/tutorial/01_first.viv -o first.c
cc -O2 first.c -lm -o first && ./first
```

## 2. Arrays

[`02_arrays.viv`](tutorial/02_arrays.viv):

```viv
# Arrays broadcast like NumPy; @ is always the matrix product.
x = linspace(0.0, 1.0, 5)
y = 2.0 * x + 1.0
show x
show y
show sum(y), mean(y)

A = [[2.0, 1.0], [1.0, 3.0]]
b = [1.0, 2.0]
s = lin_solve(A, b)
show s
show A @ s

squares = [i * i for i to 6]
show squares

test "the solve satisfies A s = b" {
    assert_allclose(A @ s, b, 1e-12)
}
```

```text
x: [0, 0.25, 0.5, 0.75, 1]
y: [1, 1.5, 2, 2.5, 3]
10  2
s: [0.2, 0.6]
[1, 2]
squares: [0, 1, 4, 9, 16, 25]
PASS the solve satisfies A s = b
```

Arithmetic between an array and a number broadcasts, functions like `sin` map
over arrays, `*` between two arrays is elementwise, and `@` is the matrix
product. The array builtins use NumPy's names (`linspace`, `mean`, `amax`,
`argsort`, `einsum`, ...); `viv features arrays` lists them.

## 3. A record

[`03_record.viv`](tutorial/03_record.viv):

```viv
# A record groups named fields; a def takes and returns one.
record Particle {
    x: Num
    v: Num
}

def step(p: Particle, dt) -> Particle {
    # one explicit Euler step of a unit-mass particle on a spring (k = 1)
    return Particle { x: p.x + dt * p.v, v: p.v - dt * p.x }
}

p = Particle { x: 1.0, v: 0.0 }
for k to 10 {
    p = step(p, 0.01)
}
show p.x, p.v

test "ten small steps stay near cos(0.1)" {
    assert_near(p.x, cos(0.1), 1e-3)
}
```

```text
0.995502  -0.09988
PASS ten small steps stay near cos(0.1)
```

A record literal names every field. Fields can be numbers, `Int`s, strings,
arrays, maps and other records. A record compiles to a C struct.

## 4. A plot

[`04_plot.viv`](tutorial/04_plot.viv):

```viv
# A figure is part of the program's output: plot prints a record that a host draws.
t = linspace(0.0, 10.0, 200)
figure("A damped oscillator", "t", "x")
plot(t, exp(-0.3 * t) * cos(2.0 * t), "x(t)")
plot(t, exp(-0.3 * t), "envelope")

test "the oscillation starts at its envelope" {
    assert_near(exp(-0.3 * t[0]) * cos(2.0 * t[0]), 1.0, 0.0)
}
```

`figure` and `plot` print structured records on standard output, in every
engine. The browser draws them; on the command line, `viv plot` turns them into
SVG:

```sh
viv run -q docs/tutorial/04_plot.viv | viv plot -o damped.svg
```

The SVG is the same bytes every time, so a test can compare a committed figure.

## 5. Steps and the explorer

[`05_steps.viv`](tutorial/05_steps.viv):

```viv
# Named steps give the explorer and `viv graph` the phases of a program.
def relax(u) {
    step smooth "average each interior point with its neighbors" {
        for i from 1 to len(u) - 1 {
            u[i] = 0.5 * (u[i - 1] + u[i + 1])
        }
    }
    return u
}

def solve_laplace(n) {
    u = zeros(n)
    u[n - 1] = 1.0
    for sweep to 500 {
        u = relax(u)
    }
    return u
}

u = solve_laplace(6)
show u

test "the relaxed profile is close to the straight line" {
    assert_allclose(u, linspace(0.0, 1.0, 6), 1e-3)
}
```

```text
u: [0, 0.2, 0.4, 0.6, 0.8, 1]
PASS the relaxed profile is close to the straight line
```

A `step` names a phase of a def and says what it computes. `viv graph` prints
the program's defs and steps with what each step reads and writes:

```text
$ viv graph docs/tutorial/05_steps.viv
relax(u)
  step smooth  "average each interior point with its neighbors"
    in:  u
    out: u
```

`viv graph --svg -o graph.svg` draws it. The explorer at
[viv-lang.org/explore](https://viv-lang.org/explore) runs a program in the page
and places each moment of the run in this graph, step by step.

`viv atlas` maps a whole program: every def the top level reaches, by module,
with the calls between them and how many times each ran in one run.

```text
$ viv atlas lib/minpack/story_rosenbrock.viv
wrote story_rosenbrock.atlas.html (516 KB): open it in a browser; it needs no network
$ viv atlas lib/minpack/story_rosenbrock.viv --text
atlas of lib/minpack/story_rosenbrock: 23 defs in 8 modules, reached from the top level (39 defs in the program)
run: 778 calls into 23 defs (one run, as the explorer runs it)
...
call tree from the top level, with the run's counts
  <main>
    mgh_start  x2
    lmder1_maxfev  x1
    lmder1  x1
      lmder1_maxfev  x1  (see above)
      lmder  x1
        enorm  x77
        qrfac  x16
...
```

The HTML file holds the data and its viewer, so it can be mailed or opened
offline. `--json` gives the same data to a tool, and `--no-run` skips the run
for a program that takes long or reads input. A run that goes on past 1e8
interpreter steps stops, and the atlas shows the calls the text makes;
`VIV_STEP_BUDGET` raises the limit.

## 6. Ask the catalog

`viv features` lists everything the language has, by area: syntax, types,
arrays, autodiff, correctness, lenses, tools, performance, libraries and the
builtins. Each entry has what it does, when to use it, an example (run as a
test in all three engines) and the source that implements it.

```sh
viv features                    # the areas and their entries
viv features records            # one entry
viv features --search eigen     # search by words
viv features --usage docs/tutorial/*.viv
```

`--usage` reads your files and lists hand-written code that a feature would
replace, such as an element loop where a whole-array expression or a reduction
does the same.

## 7. Port a small Python function

`viv` reads numeric Python with NumPy through py2viv, a translator that either
produces a program that behaves as CPython does or refuses, naming each
construct it cannot translate with its line and what to write instead.
[`06_rms.py`](tutorial/06_rms.py):

```python
import numpy as np

def rms(x):
    return np.sqrt(np.mean(x * x))

x = np.array([3.0, 4.0])
print("rms", rms(x))

def test_rms():
    assert abs(rms(x) - np.sqrt(12.5)) < 1e-12
```

```text
$ viv run -q docs/tutorial/06_rms.py
rms 3.5355339059327378
PASS test_rms
```

`def test_...` functions become `test` blocks. `viv 06_rms.py -o rms.c`
compiles the translation to C like any Viv program. The subset py2viv accepts
is in [`tools/py2viv/SUBSET.md`](../tools/py2viv/SUBSET.md).

## Where next

- The [language reference](https://viv-lang.org/reference) and the
  [browser tour](https://viv-lang.org/tour).
- [`lang/viv/examples/`](../lang/viv/examples/) and [`pde/`](../pde/): complete
  programs, each with its tests.
- [KNOWN_LIMITATIONS.md](../KNOWN_LIMITATIONS.md) before you start a port.
