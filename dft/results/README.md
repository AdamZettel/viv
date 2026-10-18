# Recorded runs

Program output kept under version control, for the runs too long to repeat at
every site build.

| File | Command | When it changes |
|---|---|---|
| `cu.out` | `dft/run.sh cu it > dft/results/cu.out` | whenever `dft/cu.viv`, `dft/cu_it_check.viv` or the engine does |

`cu.meta.json` records how long that run took and on what (169 s, 458 MB, one
core of an Apple M2 Pro) -- the one thing the program's own output cannot say.

`cu.out` is the literal standard output of the compiled program: the SCF
energies, the shown quantities, every `PASS` line, and the `plot:` lines of the
band structure. CI (`dft/check.sh cu`) reruns the same program and fails on any
`FAIL`, so a recorded run cannot drift from a passing one unnoticed for long —
but it is a recording, and the site says so where it uses it.
