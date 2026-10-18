---
title: A line plot
order: 1
output: differs
reason: Matplotlib writes a PNG and the Python side prints that it saved it; Viv prints the figure as text records (figure, plot, style, scatter), which a page, a notebook or `viv plot` draws. Those lines are left out of the comparison; the peaks line is compared.
py_drop: ^saved
viv_drop: ^(figure|plot|style|scatter):
---
- `figure("title", "xlabel", "ylabel")` replaces `plt.subplots()` and the three
  `set_` calls. The figure is drawn by whatever reads the program's output:
  on this site, the page draws it; on the command line,
  `viv run damped.viv | viv plot -o damped.svg` writes an SVG.
- `plot(t, x, "label")` is `ax.plot(t, x, label=...)`, and every series has a
  label, which is also its legend entry. A dashed line is a separate call,
  `style("envelope", "dash")`, where matplotlib takes a format string `"--"`.
- Markers without a line are `scatter(...)`, where matplotlib uses
  `ax.plot(..., "o")`.
- `np.linspace(0, 10, N)` is `linspace(0.0, 10.0, N)`, and `np.exp` and
  `np.cos` of an array are `exp` and `cos` of an array.
- Selecting the peaks with a comprehension and fancy indexing (`t[peaks]`)
  becomes one loop that appends to two arrays.
- There is no `savefig`: a Viv program's figure is its output.
