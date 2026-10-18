# lib/power against its references

Written by `lib/power/compare.py` (do not edit). Differences are the largest over all buses, generators or branches: Vm in per unit, Va in degrees, P and Q in MW and MVAr.

## MATPOWER's own solution of t_case9_pf

Reference: `lib/t/soln9_pf.mat` (AC) and `soln9_dcpf.mat` (DC) from the MATPOWER repository, commit 80b6e16. FDXB is held to the mismatch tolerance (1e-8 per unit), not to rounding.

| method | iterations | Vm | Va | Pg | Qg | Pf | Qf | Pt | Qt |
|---|---|---|---|---|---|---|---|---|---|
| nr | 4 | 4.4e-16 | 3.0e-14 | 1.7e-13 | 7.1e-13 | 1.7e-13 | 7.4e-13 | 1.7e-13 | 7.4e-13 |
| fdxb | 8 | 1.0e-09 | 8.0e-09 | 3.3e-08 | 8.2e-07 | 5.7e-08 | 8.2e-07 | 5.2e-08 | 7.9e-07 |
| dc | 1 | 0.0e+00 | 5.1e-15 | 5.7e-14 | 0.0e+00 | 5.7e-14 | 0.0e+00 | 5.7e-14 | 0.0e+00 |

## Every case against the independent SciPy power flow

Reference: `lib/power/oracle/pf_scipy.py` (MATPOWER's matrix formulas in scipy.sparse, solved by SuperLU). NR and FDXB are compared with its Newton solution, DC with its DC solution. `mismatch` is the largest power mismatch at Viv's answer, recomputed in NumPy (per unit). Times are the best of several runs on an Apple M2 Pro with other jobs running (load about 10).

| case | buses | method | iter (Viv / SciPy) | Vm | Va | Pg | Qg | Pf | Qf | mismatch | Viv ms | SciPy ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| case9 | 9 | nr | 4 / 4 | 4.4e-16 | 1.4e-14 | 1.3e-13 | 7.4e-13 | 1.3e-13 | 7.4e-13 | 1.8e-14 | 0.15 | 5.29 |
| case9 | 9 | fdxb | 6 / 4 | 5.8e-10 | 6.8e-08 | 5.9e-07 | 4.6e-07 | 5.9e-07 | 4.6e-07 | 1.0e-08 | 0.04 |  |
| case9 | 9 | dc | 1 / 1 | 0.0e+00 | 3.6e-15 | 4.3e-14 | 0.0e+00 | 4.3e-14 | 0.0e+00 |  | 0.03 | 0.46 |
| case14 | 14 | nr | 2 / 2 | 8.9e-16 | 7.1e-15 | 1.4e-13 | 6.2e-13 | 1.4e-13 | 3.8e-13 | 1.3e-10 | 0.05 | 3.18 |
| case14 | 14 | fdxb | 6 / 2 | 2.3e-10 | 4.2e-08 | 4.4e-07 | 1.9e-07 | 3.5e-07 | 1.3e-07 | 5.6e-09 | 0.05 |  |
| case14 | 14 | dc | 1 / 1 | 0.0e+00 | 8.9e-15 | 2.8e-14 | 0.0e+00 | 6.7e-14 | 0.0e+00 |  | 0.03 | 0.46 |
| case30 | 30 | nr | 3 / 3 | 8.9e-16 | 7.9e-14 | 1.2e-12 | 7.1e-13 | 8.9e-13 | 7.4e-13 | 9.6e-10 | 0.11 | 4.43 |
| case30 | 30 | fdxb | 11 / 3 | 2.4e-10 | 4.1e-08 | 5.5e-09 | 2.3e-07 | 2.3e-07 | 1.4e-07 | 4.7e-09 | 0.08 |  |
| case30 | 30 | dc | 1 / 1 | 0.0e+00 | 1.8e-15 | 0.0e+00 | 0.0e+00 | 6.4e-14 | 0.0e+00 |  | 0.04 | 0.48 |
| case118 | 118 | nr | 3 / 3 | 6.7e-16 | 2.2e-13 | 4.7e-12 | 7.7e-12 | 2.5e-12 | 7.8e-12 | 1.5e-12 | 0.31 | 4.99 |
| case118 | 118 | fdxb | 8 / 3 | 1.9e-11 | 8.3e-09 | 6.2e-08 | 6.2e-08 | 6.3e-08 | 5.9e-08 | 1.6e-09 | 0.21 |  |
| case118 | 118 | dc | 1 / 1 | 0.0e+00 | 7.1e-14 | 1.5e-12 | 0.0e+00 | 8.0e-13 | 0.0e+00 |  | 0.10 | 0.61 |
| case300 | 300 | nr | 5 / 5 | 3.3e-15 | 1.0e-12 | 2.3e-11 | 4.2e-11 | 4.4e-11 | 9.4e-11 | 1.4e-12 | 1.80 | 10.95 |
| case300 | 300 | fdxb | 9 / 5 | 3.5e-10 | 8.5e-08 | 1.2e-06 | 4.5e-07 | 1.2e-06 | 4.5e-07 | 8.2e-09 | 0.57 |  |
| case300 | 300 | dc | 1 / 1 | 0.0e+00 | 6.4e-13 | 1.0e-11 | 0.0e+00 | 1.0e-11 | 0.0e+00 |  | 0.24 | 1.04 |
| case2383wp | 2383 | nr | 6 / 6 | 1.8e-13 | 3.5e-11 | 5.9e-10 | 5.5e-10 | 4.1e-10 | 5.4e-10 | 5.7e-12 | 22.71 | 57.02 |
| case2383wp | 2383 | fdxb | 18 / 6 | 5.2e-11 | 9.2e-08 | 2.5e-07 | 4.9e-07 | 6.5e-07 | 4.9e-07 | 5.3e-09 | 6.77 |  |
| case2383wp | 2383 | dc | 1 / 1 | 0.0e+00 | 1.9e-11 | 1.1e-09 | 0.0e+00 | 2.9e-10 | 0.0e+00 |  | 2.45 | 4.10 |
| case9241pegase | 9241 | nr | 6 / 6 | 2.6e-13 | 5.0e-11 | 2.1e-09 | 7.6e-10 | 4.6e-10 | 6.5e-10 | 2.1e-09 | 134.82 | 202.22 |
| case9241pegase | 9241 | fdxb | 14 / 6 | 2.2e-11 | 7.9e-09 | 8.4e-08 | 9.2e-08 | 2.6e-07 | 9.3e-08 | 5.4e-09 | 44.98 |  |
| case9241pegase | 9241 | dc | 1 / 1 | 0.0e+00 | 7.0e-11 | 2.0e-09 | 0.0e+00 | 5.8e-10 | 0.0e+00 |  | 15.73 | 17.10 |

## The sparse LU alone

Each case's Newton Jacobian at the starting voltages, solved by `splu` (minimum-degree order, threshold 0.001) and by `scipy.sparse.linalg.splu` (SuperLU, MMD_AT_PLUS_A order, the same threshold). Fill is the entries of L and U, the diagonal counted once; SuperLU's default COLAMD order is shown too. Times include the ordering. SuperLU's time on case9241pegase varied from 22 ms to 2.5 s between runs on the loaded machine; Viv's stayed between 38 and 96 ms.

| case | n | nnz(J) | nnz(L+U) Viv | SuperLU MMD | SuperLU COLAMD | max dx / max x | residual | Viv ms | SuperLU ms |
|---|---|---|---|---|---|---|---|---|---|
| case9 | 14 | 82 | 106 | 102 | 106 | 6.4e-16 | 8.1e-16 | 0.03 | 0.02 |
| case14 | 22 | 146 | 162 | 162 | 222 | 1.2e-16 | 7.6e-16 | 0.03 | 0.03 |
| case30 | 53 | 361 | 443 | 429 | 555 | 6.6e-16 | 3.3e-15 | 0.05 | 0.04 |
| case118 | 181 | 1051 | 1343 | 1335 | 1907 | 1.1e-15 | 6.4e-15 | 0.12 | 0.10 |
| case300 | 530 | 3736 | 5546 | 5562 | 8112 | 4.8e-15 | 2.3e-13 | 0.50 | 0.36 |
| case2383wp | 4438 | 27874 | 45388 | 45045 | 73863 | 4.9e-13 | 2.0e-12 | 6.43 | 3.78 |
| case9241pegase | 17036 | 129412 | 221310 | 218729 | 365778 | 1.7e-13 | 6.8e-13 | 77.76 | 2470.30 |

## The voltages stored in the case files

Not a reference: each file stores the voltages it was distributed with (case9 a flat start; the IEEE cases the solution of their original data files, rounded; the others the voltages their authors stored), so they differ from MATPOWER's solution by up to these amounts.

| case | max dVm | max dVa (deg) |
|---|---|---|
| case9 | 4.00e-02 | 9.28e+00 |
| case14 | 1.33e-03 | 1.71e-02 |
| case30 | 3.94e-02 | 3.96e+00 |
| case118 | 1.73e-02 | 3.03e-01 |
| case300 | 8.54e-03 | 9.76e+00 |
| case2383wp | 1.25e-01 | 1.16e+01 |
| case9241pegase | 2.84e-02 | 1.36e+01 |

## Interconnection studies (lib/power/raw, contingency, study, cpf, shortcircuit)

`~/.venvs/power/bin/python lib/power/compare_study.py raw n1 study cpf sc`
with pandapower 3.5.6, PYPOWER 5.1.21 and ANDES 2.0.0; the numbers are in
compare_study.json. Timings were taken with the machine's load average
near 30, so they are upper bounds.

### PSS/E RAW reader against ANDES and pandapower

ANDES's sample RAW files (read from the installed package), read by
`vp.power.read_raw` and by ANDES. dY is the largest entry of |Y_vv - Y_ref|
(per unit); dV the largest |V_vv - V_ref| of the complex bus voltages after
the power flow, every angle measured from the swing bus. "pandapower" is
pandapower's power flow on ANDES's conversion of the same file
(andes.interop.pandapower.to_pandapower).

| case | rev | buses | branches | dY ANDES | dY pandapower | dV ANDES | dV pandapower |
|---|---|---|---|---|---|---|---|
| ieee14 | 32 | 14 | 20 | 7.1e-15 | 4.0e-15 | 1.5e-08 | 2.8e-12 |
| wscc9 | 33 | 9 | 9 | 4.4e-16 | 7.1e-15 | 2.4e-08 | 4.3e-12 |
| kundur | 32 | 10 | 15 | 1.8e-15 | 2.9e-14 | 8.1e-07 | 7.5e-16 |
| ieee39 | 33 | 39 | 46 | 1.1e-13 | 1.1e-13 | 2.9e-07 | 6.5e-15 |
| npcc | 32 | 140 | 233 | 1.1e-13 | 1.1e-13 | 4.6e-07 | 4.0e-15 |
| wecc | 32 | 179 | 263 | 4.5e-13 | 2.3e-13 | 9.6e-06 | 1.7e-11 |
| N44_BC (Nordic 44) | 33 | 44 | 79 | (see below) | | | |

ANDES's own voltages differ by up to 1e-5 because its Newton iteration
stops at its own tolerance; pandapower, solving ANDES's network to 1e-8
MVA, agrees to 2e-11. Nordic 44 has SBASE = 1000 MVA: ANDES 2.0.0 reads its
branch impedances as if on a 100 MVA base (Line.Sn = 100), so its
admittance matrix is a tenth of the file's and its power flow does not
converge; vp.power's converges in 3 iterations. No reference here reads
that file correctly, so it is not counted as checked.

MATPOWER's case14, case118 and case300 written as RAW 33 and 34 by
lib/power/oracle/write_raw.py, rotating the transformers through CW 1, 2,
3 and CZ 1, 2, 3 (one with WINDV2 = 1.02), half of each bus shunt as the
constant-admittance part of a load, a quarter of each load as constant
current and a transformer's charging as its magnetizing admittance, read
back to the .m file's admittance matrix within 1e-12 (case14) and 1e-9
(case118, case300), and to its Newton voltages within the same
(lib/power/raw.viv's tests).

### N-1 contingency analysis

vp.power.contingency (every branch and generator in service, tol 1e-10)
against a brute-force loop of PYPOWER's runpf (one outage at a time, tol
1e-10) and pandapower's run_contingency on from_ppc's network. dS is the
largest difference in branch |S| (MVA, at the more loaded end) and dVm in
bus voltage, over the outages both solve; PYPOWER cannot solve an islanded
network, so those outages are compared with pandapower only, which takes
out islanded buses as vp.power does. The pandapower columns compare the
per-bus lowest and highest Vm over the base case and every outage.

| case | outages | islanding | compared | dS PYPOWER | dVm PYPOWER | dVmin pandapower | dVmax pandapower | vp.power s | PYPOWER s |
|---|---|---|---|---|---|---|---|---|---|
| case30 | 47 | 3 | 45 | 6.4e-09 | 3.6e-11 | 1.8e-09 | 1.2e-11 | 0.01 | 0.25 |
| case118 | 240 | 9 | 232 | 1.4e-08 | 1.6e-11 | 1.0e-09 | 1.2e-10 | 0.08 | 1.53 |
| case300 | 480 | 89 | 369 | 1.7e-08 | 6.6e-11 | (0.11) | (0.11) | 0.75 | 5.43 |

On case300, 26 outages do not converge in vp.power; PYPOWER fails on the
same 23 that do not island and on none that vp.power solves. pandapower's
from_ppc network for case300 differs from MATPOWER's already in the base
case (Vm at bus 17 is 0.958 against MATPOWER's and PYPOWER's 1.065), so
the 0.11 there is pandapower's conversion, not the contingency analysis.

The DC screen: the post-outage flows estimated by LODF equal a full DC
re-solve of every non-islanding branch outage of case30 to 1e-9 MW
(lib/power/contingency.viv's tests).

### The study's transfer limit

vp.power.interconnect's limit (bisection to 2 MW), held to PYPOWER: the
case with the project at the limit, solved by PYPOWER's runpf for every
outage, has no violation (contingency, kind, element) the case without it
lacks; at the smallest output the bisection found violating, it has one.

| case | POI bus | limit MW | new at limit (PYPOWER) | new above (PYPOWER) | vp.power s |
|---|---|---|---|---|---|
| case30 | 30 | 1.56 | 0 | 2 | 0.05 |
| case118 | 38 | 971.19 | 0 | 1 | 0.93 |
| case118 | 87 | 209.47 | 0 | 1 | 1.96 |

### PV curve (continuation power flow)

The nose of the curve for a transfer into the POI (taken out at the
reference bus) by vp.power.cpf (nose_tol 1e-4), by PYPOWER's runcpf
(pseudo-arclength, step 0.02, full trace; its predictor needs a 1 x 1
array where it passes a scalar to SciPy 1.13+, patched in the harness),
and as the largest transfer at which PYPOWER's Newton power flow still
converges (bisection to 1e-3 MW, warm start). Every point of the vp.power
curve, re-solved by PYPOWER at its transfer, agrees in Vm to dVm. The last
column is the Jacobian's smallest singular value at the nose over its
value in the base case.

| case | POI | points | vp.power nose MW | PYPOWER runcpf MW | Newton limit MW | dVm points | sigma ratio |
|---|---|---|---|---|---|---|---|
| case9 | 5 | 53 | 599.851 | 599.839 | 599.851 | 2.2e-16 | 4.9e-06 |
| case30 | 30 | 50 | 240.725 | 240.716 | 240.724 | 2.2e-16 | 5.3e-05 |
| case118 | 38 | 71 | 1990.687 | 1990.686 | 1990.687 | 3.3e-16 | 8.6e-05 |

### Short circuit (IEC 60909, three-phase)

Ik'' at every bus against pandapower.shortcircuit.calc_sc (fault 3ph, case
max) on from_ppc's network, every generator a machine with the RAW file's
source impedance (cos phi 0.85), transformers given the system base as
their rating so that K_T reads their per-unit reactance as vp.power does.

| case | buses | Ik'' range kA | unit-ratio transformers | max rel. diff | same K_T rule |
|---|---|---|---|---|---|
| case118_v33.raw | 118 | 2.57 to 18.40 | 2 | 1.3e-02 | 3.0e-15 |
| ieee14.raw (ANDES) | 14 | 1.73 to 16.44 | 0 | 1.3e-15 | 1.3e-15 |
| ieee39.raw (ANDES) | 39 | 8.43 to 6973 | 1 | 1.1e-02 | 1.4e-15 |

from_ppc reads a transformer whose ratio is exactly 1 as a plain
impedance and so applies no K_T to it; vp.power applies K_T to every
branch with a ratio set, which is every RAW transformer. With those
branches given ratio 0 the two agree to rounding.
