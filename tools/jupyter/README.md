# Viv in Jupyter

A Jupyter kernel for Viv, and a converter between Viv notebooks (`.viv`) and
Jupyter notebooks (`.ipynb`).

## What a Viv notebook is

A Viv notebook is an ordinary `.viv` program. A line that is exactly `###` opens
a prose block, which runs to the next `###` line; the compiler skips it. In a
post, the first such block is the header (`key: value` lines, then prose). Every
later block is a prose cell, written in Markdown with `$math$` and `$$math$$`.
The code between two prose blocks is a code cell.

Running cell N runs cells 1 to N as one program and shows what cell N printed.
So a cell can use anything defined above it, and the whole file still runs with
`viv run` and through the C backend with the same output.

## Install the kernel

Build the compiler, then install the kernel spec for your user:

```sh
cargo build --release -p viv
python tools/jupyter/install.py
jupyter kernelspec list        # shows "viv"
```

The kernel needs `ipykernel` in the Python that runs `install.py` (Anaconda and
JupyterLab installs have it; otherwise `pip install ipykernel`). The spec points
at this checkout: `tools/jupyter` for the kernel's code and
`target/release/viv` for the compiler. Run `install.py` again after moving the
repository. `--viv PATH` uses another viv binary, and `--engine native` makes
cells compile through C unless they say otherwise.

## Open a notebook

Convert the `.viv` file, then open the `.ipynb`:

```sh
python tools/jupyter/vivnb.py tools/jupyter/examples/heat-rod.viv    # writes heat-rod.ipynb
jupyter lab tools/jupyter/examples/heat-rod.ipynb
```

In JupyterLab, pick the "Viv" kernel if it does not start on its own (the
converted notebook names it). In VS Code, open the `.ipynb`, click "Select
Kernel", then "Jupyter Kernel...", then "Viv". VS Code reads the same kernel
specs as Jupyter.

The `.viv` file is the source of truth. After editing in Jupyter, write it back:

```sh
python tools/jupyter/vivnb.py tools/jupyter/examples/heat-rod.ipynb  # writes heat-rod.viv
```

Converting `.viv` to `.ipynb` and back gives the same bytes. The post header
becomes two cells: a raw cell with the `key: value` lines, and a Markdown cell
with the header's prose. Two code cells with no Markdown cell between them
cannot stay separate in a `.viv` file, so they are joined with a blank line
between them.

`python tools/jupyter/vivnb.py --check FILE.viv ...` converts each file both
ways and reports any that do not come back unchanged.

## What a cell shows

- Text the cell prints: `show` lines, the PASS and FAIL lines of the tests in
  the cell, and `narrate` lines (as Markdown). FAIL lines go to stderr, so
  Jupyter shows them in red. A failing test is output, not an error: later
  cells still run.
- Figures, as inline SVG: `plot` (lines), `scatter`, `bars` and `plot2d`
  (a heatmap; for an animation, its last frame). They group into charts the
  same way they do on the site. Animations, 3-D scenes and vector fields
  print a one-line note instead, since they are drawn only on the site.
- A compile or runtime error, as a Jupyter error. Where the compiler can
  point at the mistake, the location is counted from the cell's own first
  line:

  ```
  error: unknown variable `zz`
    --> this cell, line 2, column 6
     2 | show zz
              ^^ not a parameter or local in scope
  ```

  A cell that fails is not added to the program, so the next cell runs as if
  it had not been there.

## Choosing the engine

The interpreter runs cells by default: it starts at once. A comment on a
cell's first line can change that:

| First line        | Effect                                                  |
| ----------------- | ------------------------------------------------------- |
| `#% native`       | this cell compiles through the C backend and runs natively |
| `#% interp`       | this cell uses the interpreter                           |
| `#% engine native` | every later cell compiles through C, until `#% engine interp` |
| `#% reset`        | forget every cell run so far                             |

These are comments to the compiler, so the file still runs as a program. A
native run compiles the whole prefix with `cc -O2`, which takes a moment, and
reuses the binary when the same cells run again.

## Running cells again

JupyterLab and VS Code tell the kernel which cell is running. Running a cell
again, edited or not, drops it and every cell run after it, then runs it after
the cells above it. That keeps a cell from running twice in the program. A
front end that sends no cell id (such as `nbconvert`) gets a simpler rule:
running the same source as the last cell again replaces it, and anything else
is added at the end.

## Tests

```sh
python -m pytest tools/jupyter/tests
```

The tests round-trip every `.viv` file in the repository; check that the cells
of each example add up to the whole program's output in both engines; install
the kernel into a scratch directory (not your user's kernels); execute the
three example notebooks with `jupyter nbconvert --to notebook --execute` and
compare every cell's output; and check that a failing test shows as a FAIL line,
a compile error as an error, and that re-running an edited cell replaces it.
