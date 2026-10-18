# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The one table of vipyr: every public function, where its program lives,
and what the docs page shows beyond what the program's manifest says.

Parameters, their types, ranges, defaults and help, the results and their
meanings, and the reference a program is checked against come from the
program itself: its `<name>.export.toml` manifest and the `#:` and `#|`
lines `viv export` reads into the wrapper's docstring (vipyr/_spec.py). This
table adds only what no manifest holds: the Python name, the field it is
listed under, renamed keywords, a runnable example, and the walk-through
slug on viv-lang.org/explore.

kind:
  program   a manifest of a whole program (`[params]`, `[results]`); the
            call is one run, the result carries the program's own tests
  kernel    one `[kernels.<name>]` entry of a kernel-collection manifest
  inferred  a top-level def of a .viv file without a manifest, exported by
            `viv export --infer`; `checked` says what it is held to
  feature   an entry of the feature catalog (catalog/catalog.py), called on
            a pandas DataFrame
  python    a Python function of vipyr (`impl`, in vipyr/_ode.py) that writes
            and compiles a Viv driver per problem; `params` and `results`
            are listed here, the prose is in the docs file; `map` names the
            Viv library def the driver calls (`file:def`), for its code map
  coming    listed, not built: calling it raises NotImplementedError
"""

# The distributions. vipyr is the core: the tools and the broadly useful
# numerics. Each other one is a directory under packages/ (or ports/) with
# its own pyproject.toml, and installs with `pip install vipyr[<extra>]`.
DISTS = {
    "vipyr": dict(path="python", module="vipyr", extra=None, license="MIT OR Apache-2.0"),
    "vipyr-chem": dict(path="packages/vipyr-chem", module="vipyr_chem", extra="chem", license="MIT OR Apache-2.0"),
    "vipyr-nuclear": dict(path="packages/vipyr-nuclear", module="vipyr_nuclear", extra="nuclear",
                          license="MIT OR Apache-2.0 (the shell model follows BIGSTICK, MIT: NOTICE)"),
    "vipyr-plasma": dict(path="packages/vipyr-plasma", module="vipyr_plasma", extra="plasma", license="MIT OR Apache-2.0"),
    "vipyr-xc": dict(path="packages/vipyr-xc", module="vipyr_xc", extra="xc", license="MPL-2.0 (Libxc's)"),
    "vipyr-symreg": dict(path="packages/vipyr-symreg", module="vipyr_symreg", extra=None, license="MIT OR Apache-2.0"),
    "vipyr-hardware": dict(path="packages/vipyr-hardware", module="vipyr_hardware", extra=None,
                           license="MIT OR Apache-2.0"),
    "vipyr-liquids": dict(path="ports/liquids", module="vipyr_liquids", extra=None,
                          license="CC0-1.0 (Allen and Tildesley's examples are CC0)"),
}

# family -> the distribution that carries it
FAMILY_DIST = {
    "special": "vipyr", "linalg": "vipyr", "finance": "vipyr", "features": "vipyr", "statmech": "vipyr",
    "randla": "vipyr", "control": "vipyr", "ode": "vipyr", "optimize": "vipyr", "gpt": "vipyr", "gguf": "vipyr", "stats": "vipyr", "power": "vipyr",
    "scf": "vipyr-chem", "dft": "vipyr-chem",
    "nuclear": "vipyr-nuclear", "plasma": "vipyr-plasma", "xc": "vipyr-xc",
    "symreg": "vipyr-symreg", "hardware": "vipyr-hardware", "liquids": "vipyr-liquids",
}

FIELDS = [
    ("scf", "Quantum chemistry", "Hartree-Fock for molecules in Gaussian basis sets."),
    ("dft", "Electronic structure of solids", "Plane-wave density-functional theory of crystals."),
    ("nuclear", "Nuclear structure", "Mean-field and shell-model calculations of nuclei."),
    ("plasma", "Plasma physics", "Kinetic simulations of a collisionless plasma."),
    ("special", "Special functions", "Functions with no closed form that physics codes call in their inner loops."),
    ("linalg", "Linear algebra", "The Euclidean norm, and SciPy-named aliases of vp.control's solvers."),
    ("control", "Control", "LQR gains, Riccati and Lyapunov solvers, and closed-loop simulation of a cart-pole."),
    ("ode", "Differential equations", "Stiff and non-stiff initial value problems: Dormand-Prince, Radau IIA "
     "and BDF, with events, exact Jacobians and batches."),
    ("optimize", "Fitting and optimization", "Nonlinear least squares, curve fitting with parameter "
     "covariances and root finding by MINPACK's Levenberg-Marquardt, the residuals compiled with it."),
    ("gpt", "Language models", "A small GPT (nanoGPT's shape, the speedrun's Muon and Adam) trained on text, "
     "sampled, and returned as named arrays."),
    ("gguf", "Model files", "GGUF files, the format of ggml and llama.cpp: metadata, the tensor table, and "
     "tensors dequantized from ggml's number formats."),
    ("stats", "Statistics", "Probability distributions, the special functions under them, descriptive "
     "statistics, least squares with inference, and generalized linear models."),
    ("finance", "Option pricing", "Black-Scholes, binomial trees, Heston by Fourier-cosine series, finite differences."),
    ("features", "Features on event tables", "Per-entity features with state, called on a pandas DataFrame."),
    ("statmech", "Statistical mechanics", "Lattice models sampled by Monte Carlo, checked against exact solutions."),
    ("randla", "Randomized linear algebra", "Low-rank approximations and trace estimates from random sketches."),
    ("xc", "Exchange-correlation functionals", "Libxc's LDA and GGA functionals, translated to Viv."),
    ("symreg", "Symbolic regression", "Formulas recovered from data."),
    ("power", "Power systems", "AC and DC power flow on MATPOWER cases: Newton-Raphson, fast-decoupled and DC."),
    ("hardware", "Hardware simulation", "Logic and processors simulated gate by gate or instruction by instruction."),
    ("liquids", "Simulation of liquids", "Molecular dynamics and Monte Carlo of the Lennard-Jones liquid, from Allen "
     "and Tildesley's examples, with g(r), diffusion and the equation of state they are tested against."),
]

# One paragraph per family for its index page (/vipyr/<family>). A family
# without one shows its one-line description only.
OVERVIEWS = {
    # Each overview: one plain line on what the family is for, then "- " list
    # lines, then a line on what it is checked against (rendered by
    # www/src/pages/vipyr/[family]/index.astro).
    "scf": (
        "Hartree-Fock for molecules, with PySCF's names, shapes and units.\n"
        '- give a molecule, a Gaussian basis set and RHF or UHF\n'
        '- get back the energy, the orbitals and the density\n'
        "It is a Viv port of PySCF's SCF, checked against PySCF 2.5.0 on every case of the port."
    ),
    "dft": (
        'Plane-wave density-functional theory for crystals.\n'
        '- name the material, the plane-wave cutoff and the k-point mesh\n'
        '- get back the total energy, the band energies and the band gap\n'
        'Checked against an independent NumPy implementation and published band energies.'
    ),
    "nuclear": (
        'Nuclear structure two ways.\n'
        '- Skyrme Hartree-Fock, the mean field of a spherical nucleus\n'
        '- the shell model, configuration interaction in a valence space\n'
        'The shell model is checked against BIGSTICK.'
    ),
    "plasma": (
        'A Vlasov-Poisson solver that measures the Landau damping rate of a Langmuir wave.\n'
        'The rate is compared with the root of the plasma dispersion relation.'
    ),
    "statmech": (
        'The two-dimensional Ising model, sampled by Monte Carlo.\n'
        '- Metropolis or Wolff updates\n'
        "- each run reports the exact values it can be compared with: Onsager's critical temperature and infinite-lattice energy, and the finite lattice's exact energy"
    ),
    "randla": (
        'Low-rank factorizations and trace estimates of a matrix, computed from random sketches of it.\n'
        'Each result is checked against NumPy, SciPy or scikit-learn.'
    ),
    "special": (
        'Special functions that physics codes evaluate in their inner loops.\n'
        '- the Boys function of Gaussian integrals\n'
        '- the plasma dispersion function\n'
        '- the Faddeeva function'
    ),
    "linalg": (
        "Dense linear algebra.\n"
        '- `enorm`, the Euclidean norm without overflow or underflow\n'
        "- `schur`, `solve_care`, `solve_lyapunov` and `lqr`, SciPy's names for the solvers of vp.control\n"
        "The norm is checked against mpmath and SciPy, the solvers against SciPy and by the residuals of their equations."
    ),
    "control": (
        'Control design for the cart-pole in four calls.\n'
        '- `cartpole_model`, the cart-pole linearized about the upright\n'
        '- `lqr`, the optimal gain for the weights you choose\n'
        '- `care`, `lyapunov` and `schur`, the Riccati and Lyapunov solutions behind it\n'
        '- `simulate`, the nonlinear system run under that gain\n'
        'The solvers are those of `vp.linalg`, checked by relative residuals and against SciPy.'
    ),
    "ode": (
        "Initial value problems `y' = f(t, y, p)`, with the right-hand side written in Viv or translated from Python.\n"
        '- Dormand-Prince 5(4) for non-stiff problems, Radau IIA of order 5 and variable-order BDF for stiff ones\n'
        '- `auto`, which switches from the first to the second when it detects stiffness\n'
        '- exact Jacobians, by reverse-mode differentiation of the right-hand side\n'
        '- events located on the dense output\n'
        '- `batch`, which solves thousands of parameter sets in one call\n'
        "Dormand-Prince follows SciPy's `solve_ivp` step for step, Radau and BDF take its steps to within two, and all three are checked against the published reference solutions of the stiff test set."
    ),
    "optimize": (
        "Nonlinear least squares by MINPACK's Levenberg-Marquardt, with SciPy's names.\n"
        "- `least_squares` and `root`, as SciPy's `method='lm'`\n"
        "- `curve_fit`, which returns the parameters and their covariance as SciPy's does\n"
        '- residuals written in Viv, or in Python and translated by py2viv\n'
        "- exact Jacobians by reverse-mode differentiation, or MINPACK's forward differences\n"
        'The port follows the Fortran step for step and is checked against the certified values of all 27 NIST nonlinear regression problems.'
    ),
    "gpt": (
        'A small GPT in the shape of nanoGPT and the modded-nanogpt speedrun, trained on a CPU in seconds.\n'
        '- token and position embeddings, RMSNorm, multi-head causal attention and a ReLU-squared MLP\n'
        "- training by Muon on the block matrices and Adam on the rest, with gradients from Viv's `value_and_grad`\n"
        '- `train` returns the model as named arrays (`model.blocks[0].attn.Wo`), and `generate` and `loss` take it back, edits included\n'
        'Checked against a NumPy version, a PyTorch reimplementation and central differences.'
    ),
    "gguf": (
        'Read GGUF files, the model format of ggml and llama.cpp.\n'
        '- `read` gives the metadata and the tensor table\n'
        '- `tensor` gives one tensor as a float32 array, from F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q4_K or Q6_K\n'
        "Every number equals gguf-py 0.19.0's `dequantize` bit for bit, in all three engines."
    ),
    "stats": (
        'Probability distributions, summary statistics and regression, with the names and results of SciPy, statsmodels and R.\n'
        "- 19 of SciPy's distributions under the same names and parameters, such as `vp.stats.norm.cdf(x)` and `vp.stats.t.ppf(q, df)`\n"
        '- the special functions under them: log-gamma, digamma, the error function, and the incomplete gamma and beta functions\n'
        "- summary statistics: the mean, the variance and R's nine kinds of sample quantile\n"
        "- linear regression by least squares, with statsmodels' summary table\n"
        "- generalized linear models, fitted the way R's `glm` fits them\n"
        "The special functions are checked against mpmath, the regressions against NIST's certified results, and the GLMs against R."
    ),
    "finance": (
        'Option pricing.\n'
        '- Black-Scholes prices, Greeks by reverse-mode differentiation, and implied volatility\n'
        '- Cox-Ross-Rubinstein binomial trees\n'
        '- Heston prices by Fourier-cosine series\n'
        '- American puts by Crank-Nicolson\n'
        'Each is checked against a published value or an independent method.'
    ),
    "liquids": (
        "The Lennard-Jones programs of Allen and Tildesley's Computer Simulation of Liquids (2nd ed., 2017), ported to Viv.\n"
        '- molecular dynamics at constant energy (velocity Verlet) and constant temperature (Nosé-Hoover chain)\n'
        '- Metropolis Monte Carlo at constant volume and constant pressure\n'
        '- all-pairs, link-cell or Verlet neighbor lists\n'
        '- block averages with error bars, g(r), the velocity autocorrelation and the mean-square displacement\n'
        '- the equation of state of Thol et al.\n'
        "Every program reproduces the book's Fortran bit for bit from the same start and random numbers. If you use it, cite the book."
    ),
    "power": (
        'Steady-state power flow on transmission networks given as MATPOWER cases.\n'
        '- Newton-Raphson in polar coordinates, fast-decoupled XB, or the linear DC model\n'
        '- generator reactive limits, optionally enforced\n'
        '- a sparse admittance matrix and Jacobian, factored by a sparse LU written in Viv\n'
        "Checked against MATPOWER's solved cases and an independent SciPy power flow, from 9 to 2,383 buses."
    ),
    "features": (
        'Per-entity features on an event table, called on a pandas DataFrame.\n'
        'Each feature is checked to read no row later than the one it is computed for.'
    ),
}

# Each entry: qualified name -> fields. `example` is Python run as written by
# python/tests/test_vipyr_programs.py; it leaves its result in `r`.
FUNCTIONS = {
    # ------------------------------------------------------------ programs
    "scf.run": dict(
        kind="program", manifest="study/ports/pyscf/vivscf.export.toml", result="SCFResult",
        explore="pyscf-port-water",
        example='''import vipyr as vp
# water in STO-3G; atoms= (atomic numbers) and coords= (angstrom) for another molecule
r = vp.scf.run(basis="sto-3g", method="rhf")
print(r.e_tot, r.converged, r.mo_energy[:3])
assert r.check()''',
    ),
    "dft.pw": dict(
        kind="program", manifest="dft/pwdft.export.toml", result="PWResult",
        aliases={"kmesh": "nkmesh"}, explore="silicon-dense",
        example='''import vipyr as vp
r = vp.dft.pw(material="si")          # the deck's ecut and k-mesh
print(r.etot, r.gap, r.evb.shape)
assert r.check()''',
    ),
    "nuclear.skyrme_hf": dict(
        kind="program", manifest="lib/skyrme/skyrmehf.export.toml", result="SkyrmeResult",
        explore="skyrme-o16",
        example='''import vipyr as vp
r = vp.nuclear.skyrme_hf()             # oxygen-16 with SLy5
print(r.e_per_a, r.r_ch, r.levels)
assert r.check()''',
    ),
    "plasma.landau": dict(
        kind="program", manifest="plasma/vlasov.export.toml", result="LandauResult",
        aliases={"k": "k0"},
        example='''import vipyr as vp
r = vp.plasma.landau(setup="k05")      # k lambda_D = 0.5
print(r.gamma, r.gamma_exact, r.omega)
assert r.check()''',
    ),
    "statmech.ising": dict(
        kind="program", manifest="stat/ising/ising_mc.export.toml", result="IsingResult",
        example='''import vipyr as vp
r = vp.statmech.ising()                 # 16 x 16 at T = 2.0, Metropolis
print(r.energy, r.energy_err, r.mag_abs, r.exact_energy)
assert r.check()''',
    ),
    "liquids.md_nve": dict(
        kind="program", manifest="ports/liquids/md_nve.export.toml", result="MDResult",
        example='''import vipyr as vp
# 108 atoms at density 0.75 from an fcc lattice at T = 1, 2 blocks of 100 steps
r = vp.liquids.md_nve(n=108, nblock=2, nstep=100)
print(r.e_cs, r.p_cs, r.p_cs_err, r.t_kin, r.conserved_msd)
assert r.check()''',
    ),
    "liquids.md_nvt": dict(
        kind="program", manifest="ports/liquids/md_nvt.export.toml", result="MDResult",
        example='''import vipyr as vp
r = vp.liquids.md_nvt(n=108, temperature=1.0, nblock=2, nstep=100)
print(r.e_cs, r.e_cs_err, r.p_cs, r.t_kin, r.conserved)
assert r.check()''',
    ),
    "liquids.mc_nvt": dict(
        kind="program", manifest="ports/liquids/mc_nvt.export.toml", result="MCResult",
        example='''import vipyr as vp
r = vp.liquids.mc_nvt(n=108, temperature=1.0, nblock=2, nstep=20)
print(r.e_cut, r.e_cut_err, r.p_cut, r.move_ratio)
assert r.check()''',
    ),
    "liquids.mc_npt": dict(
        kind="program", manifest="ports/liquids/mc_npt.export.toml", result="MCResult",
        example='''import vipyr as vp
r = vp.liquids.mc_npt(n=256, pressure=0.69, nblock=2, nstep=10)
print(r.density_avg, r.e_cut, r.volume_ratio, r.box_final)
assert r.check()''',
    ),
    "liquids.eos": dict(
        kind="program", manifest="ports/liquids/eos_lj.export.toml", result="EOSResult",
        example='''import vipyr as vp
r = vp.liquids.eos(temperature=1.0, density=0.75)
print(r.p_full, r.e_full, r.cv_full)       # Thol et al. 2016: 0.3996, -3.7212, 2.2630
print(r.p_cs, r.e_cs)                      # cut and shifted at 2.5: 0.9897, -2.9287
assert r.check()''',
    ),
    "liquids.pair_distribution": dict(
        kind="program", manifest="ports/liquids/pair_distribution.export.toml", result="PairDistributionResult",
        example='''import vipyr as vp
m = vp.liquids.md_nve(n=108, nblock=1, nstep=50)
r = vp.liquids.pair_distribution(n=108, box=m.box, frames=m.r_final, dr=0.05)
print(r.r_mid[r.g.argmax()])               # the first peak, near 1.1 sigma
assert r.check()''',
    ),
    "liquids.diffusion": dict(
        kind="program", manifest="ports/liquids/diffusion.export.toml", result="DiffusionResult",
        example='''import vipyr as vp
r = vp.liquids.diffusion()                 # two atoms moving at constant velocity
print(r.vacf[:3], r.msd[:3])
assert r.check()''',
    ),
    "liquids.error_calc": dict(
        kind="program", manifest="ports/liquids/error_calc.export.toml", result="ErrorCalcResult",
        example='''import vipyr as vp
r = vp.liquids.error_calc(nstep=4096, nequil=1000, n_repeat=4)
print(r.si_exact, r.si_halving)            # s levels off toward 2 tau = 800 steps
assert r.check()''',
    ),
    "nuclear.shell_model": dict(
        kind="program", manifest="lib/nuclear/shell/shellmodel.export.toml", result="ShellModelResult",
        example='''import vipyr as vp
# neon-20: two protons and two neutrons above oxygen-16, USDB
r = vp.nuclear.shell_model(interaction="usdb", z=2, n=2, nstates=5)
print(r.energies, r.j, r.dim)
assert r.check()''',
    ),
    # ------------------------------------------------------------- kernels
    "special.boys": dict(
        kind="kernel", manifest="kernels/k1/k1.export.toml", entry="boys",
        example='''import vipyr as vp
r = vp.special.boys(n_max=8, T=2.5)
print(r.F)
assert r.check()''',
    ),
    "special.plasma_z": dict(
        kind="kernel", manifest="kernels/k1/k1.export.toml", entry="plasma_z",
        example='''import vipyr as vp
r = vp.special.plasma_z(zeta=1.0 + 0.5j)
print(r.Z, r.Zp)
assert r.check()''',
    ),
    "special.faddeeva_w": dict(
        kind="kernel", manifest="kernels/k1/k1.export.toml", entry="faddeeva_w",
        example='''import vipyr as vp
r = vp.special.faddeeva_w(z=2.0 + 1.0j)
print(r.w)
assert r.check()''',
    ),
    "linalg.enorm": dict(
        kind="kernel", manifest="kernels/k1/k1.export.toml", entry="enorm",
        example='''import numpy as np
import vipyr as vp
r = vp.linalg.enorm(x=np.array([3e200, 4e200]))
print(r.norm)
assert r.check()''',
    ),
    "linalg.schur": dict(
        kind="kernel", manifest="kernels/k1/k1.export.toml", entry="schur",
        example='''import numpy as np
import vipyr as vp
A = np.array([[1.0, 2.0, 0.0], [-2.0, 1.0, 3.0], [0.0, 1.0, 4.0]])
r = vp.linalg.schur(A=A)
T, Z = r                                  # unpacks like scipy.linalg.schur
print(np.max(np.abs(A - Z @ T @ Z.T)))
assert r.check()''',
    ),
    "linalg.solve_care": dict(
        kind="kernel", manifest="kernels/k1/k1.export.toml", entry="solve_care",
        example='''import numpy as np
import vipyr as vp
A = np.array([[0.0, 1.0], [0.0, 0.0]]); B = np.array([[0.0], [1.0]])
r = vp.linalg.solve_care(A=A, B=B, Q=np.eye(2), R=np.eye(1))
print(r.X)
assert r.check()''',
    ),
    "linalg.solve_lyapunov": dict(
        kind="kernel", manifest="kernels/k1/k1.export.toml", entry="solve_lyapunov",
        example='''import numpy as np
import vipyr as vp
A = np.array([[-1.0, 0.5], [0.0, -2.0]])
r = vp.linalg.solve_lyapunov(A=A, Q=np.eye(2))
print(np.max(np.abs(A @ r.X + r.X @ A.T - np.eye(2))))
assert r.check()''',
    ),
    "linalg.lqr": dict(
        kind="kernel", manifest="kernels/k1/k1.export.toml", entry="lqr",
        example='''import numpy as np
import vipyr as vp
A = np.array([[0.0, 1.0], [0.0, 0.0]]); B = np.array([[0.0], [1.0]])
r = vp.linalg.lqr(A=A, B=B, Q=np.eye(2), R=np.eye(1))
print(r.K)                                # [[1, sqrt 3]]
assert r.check()''',
    ),
    # ------------------------------------------------------------- control
    # The same validated kernels as linalg.*, under the names a control engineer
    # looks for, plus the cart-pole's model and closed-loop run (lib/cartpole_api.viv).
    "control.lqr": dict(
        kind="kernel", manifest="kernels/k1/k1.export.toml", entry="lqr", result="LqrResult",
        example='''import numpy as np
import vipyr as vp
A, B = vp.control.cartpole_model(h=1e-5)     # linearized about the upright
r = vp.control.lqr(A=A, B=B, Q=np.diag([1.0, 1.0, 20.0, 1.0]), R=np.array([[0.05]]))
print(np.round(r.K, 4))                      # u = -K s
print(np.linalg.eigvals(A - B @ r.K).real.max() < 0)   # the closed loop is stable
assert r.check()''',
    ),
    "control.care": dict(
        kind="kernel", manifest="kernels/k1/k1.export.toml", entry="solve_care", result="CareResult",
        example='''import numpy as np
import vipyr as vp
A = np.array([[0.0, 1.0], [0.0, 0.0]]); B = np.array([[0.0], [1.0]])
r = vp.control.care(A=A, B=B, Q=np.eye(2), R=np.eye(1))
X = r.X
print(np.abs(A.T @ X + X @ A - X @ B @ B.T @ X + np.eye(2)).max() < 1e-13)
assert r.check()''',
    ),
    "control.lyapunov": dict(
        kind="kernel", manifest="kernels/k1/k1.export.toml", entry="solve_lyapunov", result="LyapunovResult",
        example='''import numpy as np
import vipyr as vp
A = np.array([[-1.0, 0.5], [0.0, -2.0]])
r = vp.control.lyapunov(A=A, Q=np.eye(2))
print(np.abs(A @ r.X + r.X @ A.T - np.eye(2)).max() < 1e-14)
assert r.check()''',
    ),
    "control.schur": dict(
        kind="kernel", manifest="kernels/k1/k1.export.toml", entry="schur", result="SchurResult",
        example='''import numpy as np
import vipyr as vp
A, B = vp.control.cartpole_model(h=1e-5)
r = vp.control.schur(A=A)
T, Z = r                                     # the open-loop modes are on T's diagonal
print(np.round(np.sort(np.diag(T)), 4))      # one positive: the pole falls
assert r.check()''',
    ),
    "control.cartpole_model": dict(
        kind="inferred", source="lib/cartpole_api.viv", entry="cartpole_model",
        results=[("A", "Arr", "the 4 by 4 state matrix df/ds at the upright, state (x, xdot, theta, thetadot)"),
                 ("B", "Arr", "the 4 by 1 input matrix df/du, the force on the cart in newtons")],
        checked="Its own test: near the upright the nonlinear rate equals A s to 1e-10 (lib/cartpole_api.viv); "
                "the LQR gains it leads to match the cart-pole post's (posts/2026-09-22-lqr-balances-the-cart-pole.viv).",
        example='''import vipyr as vp
r = vp.control.cartpole_model(h=1e-5)
A, B = r
print(round(A[3, 2], 6), round(B[3, 0], 6))   # how fast the pole tips, and what a push does to it
assert r.check()''',
    ),
    "control.simulate": dict(
        kind="inferred", source="lib/cartpole_api.viv", entry="cartpole_simulate",
        results=[("trajectory", "Arr", "one row per time step: t, x, xdot, theta, thetadot, u (seconds, "
                  "meters, radians, newtons)")],
        checked="The cart-pole post's closed loop: from 25 degrees under its LQR gains the pole ends within "
                "0.00096 rad of upright after 6 s (posts/2026-09-22-lqr-balances-the-cart-pole.viv).",
        example='''import numpy as np
import vipyr as vp
A, B = vp.control.cartpole_model(h=1e-5)
K = vp.control.lqr(A=A, B=B, Q=np.diag([1.0, 1.0, 20.0, 1.0]), R=np.array([[0.05]])).K
r = vp.control.simulate(K=K.ravel(), s0=np.array([0.0, 0.0, np.radians(25), 0.0]), dt=0.01, steps=600)
print(r.trajectory.shape, abs(r.trajectory[-1, 3]) < 1e-3)   # 6 s later the pole is upright
assert r.check()''',
    ),
    # ---------------------------------------------------------------- power
    "power.runpf": dict(
        kind="python", module="_power", impl="runpf", docs="lib/power/power.docs.toml", result="PowerFlowResult",
        check_text="this solve converged",
        raises=[
            ("ValueError", "an unknown method, a case name that is not shipped, or a case dict without "
             "baseMVA, bus, gen and branch"),
            ("module.VivStop", "the case has no bus that can be the reference, or every generator is at a "
             "reactive limit"),
        ],
        sources=["lib/power/api.viv", "lib/power/pf.viv", "lib/power/ybus.viv", "lib/power/splu.viv",
                 "lib/power/case.viv"],
        tests=["lib/power/matpower_ref.viv"],
        params=[
            ("case", "str | dict", None, "a shipped MATPOWER case by name ('case9', 'case14', 'case30', 'case118', "
             "'case300', 'case2383wp'), a path to a MATPOWER .m file, or a dict with baseMVA, bus, gen and branch "
             "(MATPOWER's columns, as in PYPOWER's ppc)"),
            ("method", "str", "'nr'", "'nr' (Newton-Raphson, polar), 'fdxb' (fast-decoupled, XB) or 'dc'"),
            ("enforce_q_lims", "bool", "False", "hold generators at their reactive limits, turning their buses "
             "into PQ buses, and solve again until no limit is violated (AC methods only)"),
            ("tol", "float", "1e-08", "the largest power mismatch allowed, per unit (MATPOWER's pf.tol)"),
            ("max_it", "int", "None", "the most iterations: None for MATPOWER's defaults, 10 for 'nr' and 30 "
             "for 'fdxb'"),
        ],
        results=[
            ("success", "bool", "whether the flow converged"),
            ("iterations", "int", "Newton or fast-decoupled iterations (summed over the Q-limit passes)"),
            ("method", "str", "the method used"),
            ("mismatch", "float", "the largest power mismatch left, per unit (0 for 'dc')"),
            ("vm", "ndarray[float64]", "bus voltage magnitudes, per unit, in the case's bus order"),
            ("va", "ndarray[float64]", "bus voltage angles, degrees"),
            ("pg", "ndarray[float64]", "generator real outputs, MW, in the case's gen order"),
            ("qg", "ndarray[float64]", "generator reactive outputs, MVAr"),
            ("pf", "ndarray[float64]", "real power entering each branch at its from end, MW"),
            ("qf", "ndarray[float64]", "reactive power entering each branch at its from end, MVAr"),
            ("pt", "ndarray[float64]", "real power entering each branch at its to end, MW"),
            ("qt", "ndarray[float64]", "reactive power entering each branch at its to end, MVAr"),
            ("bus", "ndarray[float64]", "the bus matrix with VM and VA filled, MATPOWER's columns"),
            ("gen", "ndarray[float64]", "the gen matrix with PG and QG filled"),
            ("branch", "ndarray[float64]", "the branch matrix widened to 17 columns, PF, QF, PT, QT filled"),
            ("baseMVA", "float", "the system base, MVA"),
        ],
        checked="MATPOWER's own solution of its test case t_case9_pf (lib/t/soln9_pf.mat and soln9_dcpf.mat): "
                "every voltage, generator and branch result to 1e-12 by NR and DC, and by FDXB to its "
                "convergence tolerance; and an independent NumPy/SciPy power flow (lib/power/oracle/pf_scipy.py) "
                "on case9 to case2383wp, where the largest differences are 2e-13 per unit in Vm and 4e-11 "
                "degrees in Va by NR (lib/power/compare.py, lib/power/COMPARISON.md).",
        example='''import vipyr as vp
r = vp.power.runpf("case118")               # Newton-Raphson
print(r.success, r.iterations, r.vm.min(), r.va.max())
d = vp.power.runpf("case118", method="dc")  # the linear model
print(abs(d.va - r.va).max())               # degrees between the two
assert r.check()''',
    ),
    "power.read_raw": dict(
        kind="python", module="_power", impl="read_raw", docs="lib/power/power.docs.toml", result="RawCase",
        check_text="this solve converged",
        raises=[
            ("ValueError", "an unknown method, a case name that is not shipped, or a case dict without "
             "baseMVA, bus, gen and branch"),
            ("module.VivStop", "the case has no bus that can be the reference, or every generator is at a "
             "reactive limit"),
        ],
        sources=["lib/power/raw.viv", "lib/power/api.viv"],
        tests=["lib/power/raw.viv"],
        params=[("path", "str", None, "a PSS/E RAW file, revision 32, 33 or 34")],
        results=[
            ("baseMVA", "float", "the system base, MVA (the RAW header's SBASE)"),
            ("bus", "ndarray[float64]", "MATPOWER's bus matrix, 13 columns; loads, shunts and line-end shunts summed per bus"),
            ("gen", "ndarray[float64]", "MATPOWER's gen matrix, 21 columns"),
            ("branch", "ndarray[float64]", "MATPOWER's branch matrix, 13 columns: lines, then switching devices, then transformers"),
            ("machine", "ndarray[float64]", "per generator: MBASE, ZR, ZX (the source impedance on the machine base)"),
            ("areas", "ndarray[float64]", "per area: number, swing bus, desired interchange (MW)"),
            ("rev", "int", "the RAW revision"),
            ("success", "bool", "always True: a file that cannot be read raises"),
        ],
        checked="ANDES 2.0.0's own reader and power flow, and pandapower 3.5.6 on ANDES's conversion, on the "
                "seven RAW cases ANDES ships (IEEE 14, WSCC 9, Kundur, IEEE 39, NPCC 140, WECC 179, Nordic 44): "
                "the admittance matrices agree to 5e-13 and pandapower's voltages to 3e-11 per unit; and MATPOWER's "
                "case14, case118 and case300 written as RAW 33 and 34 in every transformer code read back to their "
                "own admittance matrix and power flow (lib/power/compare_study.py, lib/power/COMPARISON.md).",
        example='''import vipyr as vp
from vipyr import _spec
r = vp.power.read_raw(_spec.source("lib/power/cases/raw/case118_v33.raw"))
pf = vp.power.runpf({"baseMVA": r.baseMVA, "bus": r.bus, "gen": r.gen, "branch": r.branch})
print(r.rev, r.bus.shape, pf.success, pf.vm.min())
assert r.check()''',
    ),
    "power.contingency": dict(
        kind="python", module="_power", impl="contingency", docs="lib/power/power.docs.toml",
        check_text="this solve converged",
        raises=[
            ("ValueError", "an unknown method, a case name that is not shipped, or a case dict without "
             "baseMVA, bus, gen and branch"),
            ("module.VivStop", "the case has no bus that can be the reference, or every generator is at a "
             "reactive limit"),
        ],
        result="ContingencyResult",
        sources=["lib/power/contingency.viv", "lib/power/api.viv", "lib/power/pf.viv"],
        tests=["lib/power/contingency.viv"],
        params=[
            ("case", "str | dict", None, "a case as vp.power.runpf takes it, or a .raw path"),
            ("contingencies", "str | list", "'n-1'", "'n-1' (every branch and generator in service), 'branches', "
             "'gens', or a list of ('branch', k) and ('gen', g), rows of the case from 0"),
            ("limits", "dict", "None", "vmin, vmax (base case), vmin_post, vmax_post (after an outage), rate_pre, "
             "rate_post ('A', 'B' or 'C'); defaults 0.95, 1.05, 0.90, 1.10, 'A', 'B'"),
            ("screen", "float", "0.9", "the DC screen flags a post-outage flow above this fraction of rate_post"),
            ("tol", "float", "1e-08", "the power flow's mismatch tolerance, per unit"),
            ("max_it", "int", "10", "Newton iterations per solve"),
        ],
        results=[
            ("kinds", "ndarray[int64]", "per contingency: 1 a branch, 2 a generator"),
            ("elems", "ndarray[int64]", "per contingency: the branch or generator row"),
            ("converged", "ndarray[bool]", "per row (the base case first): whether the AC flow converged"),
            ("iterations", "ndarray[int64]", "per row: Newton iterations from the warm start"),
            ("islanded", "ndarray[int64]", "per row: buses left without a path to the reference"),
            ("lost_mw", "ndarray[float64]", "per row: the load on those buses, MW"),
            ("flows", "ndarray[float64]", "rows x branches: |S| at the more loaded end, MVA (NaN where no convergence)"),
            ("vm", "ndarray[float64]", "rows x buses: voltage magnitude, per unit (0 on an islanded bus)"),
            ("violations", "ndarray[float64]", "one row per violation: contingency (-1 the base case), kind (1 "
             "thermal, 2 low voltage, 3 high voltage, 4 no convergence), branch or bus row, value, limit, percent"),
            ("dc_flows", "ndarray[float64]", "contingencies x branches: the DC estimate by LODF and PTDF, MW"),
            ("dc_flagged", "ndarray[bool]", "contingencies x branches: DC flow above screen times rate_post"),
            ("ptdf", "ndarray[float64]", "branches x buses: the DC power transfer distribution factors"),
            ("lodf", "ndarray[float64]", "branches x branches: the DC line outage distribution factors"),
            ("bridges", "ndarray[int64]", "branches whose outage islands part of the network"),
            ("success", "bool", "whether the base case converged"),
        ],
        checked="PYPOWER 5.1.21's runpf, one outage at a time (brute force), on case30, case118 and case300: "
                "branch |S| to 2e-8 MVA and Vm to 7e-11 over every outage both solve, the same 23 outages of "
                "case300 diverge in both; pandapower 3.5.6's run_contingency on from_ppc's network (case30, "
                "case118): the per-bus lowest and highest Vm over all outages to 2e-9; the LODF estimate equals a "
                "full DC re-solve to 1e-9 MW (lib/power/compare_study.py, lib/power/COMPARISON.md).",
        example='''import vipyr as vp
r = vp.power.contingency("case30")
print(r.converged.all(), r.violations.shape, r.islanded.max())
assert r.check()''',
    ),
    "power.interconnect": dict(
        kind="python", module="_power", impl="interconnect", docs="lib/power/power.docs.toml",
        check_text="this solve converged",
        raises=[
            ("ValueError", "an unknown method, a case name that is not shipped, or a case dict without "
             "baseMVA, bus, gen and branch"),
            ("module.VivStop", "the case has no bus that can be the reference, or every generator is at a "
             "reactive limit"),
        ],
        result="InterconnectionStudy",
        sources=["lib/power/study.viv", "lib/power/contingency.viv", "lib/power/api.viv"],
        tests=["lib/power/study.viv"],
        params=[
            ("case", "str | dict", None, "a case as vp.power.runpf takes it, or a .raw path"),
            ("poi_bus", "int", None, "the point of interconnection, a bus number of the case"),
            ("mw", "float", None, "the project's output, MW"),
            ("mvar_range", "tuple", "None", "(qmin, qmax), MVAr: with qmin < qmax the project holds the POI's "
             "pre-project voltage; with qmin == qmax it injects that Q; None for a 0.95 power factor either way"),
            ("contingencies", "str | list", "'n-1'", "as for vp.power.contingency"),
            ("limits", "dict", "None", "as for vp.power.contingency"),
            ("sink", "str", "'slack'", "where the project's power goes: 'slack' (the reference bus) or 'gens' "
             "(every other generator in proportion to its output)"),
            ("mw_max", "float", "None", "the top of the transfer-limit search, MW; None for 4 times mw"),
            ("tol_mw", "float", "1.0", "the transfer limit's bisection stops within this many MW"),
            ("worse_pct", "float", "1.0", "a violation present before counts as worsened when its percentage "
             "rises by more than this"),
        ],
        results=[
            ("violations_before", "ndarray[float64]", "the violations without the project (rows as in "
             "vp.power.contingency's violations)"),
            ("violations_after", "ndarray[float64]", "the violations with it"),
            ("change", "ndarray[int64]", "per row of violations_after: 1 new, 2 worsened, 0 neither"),
            ("new", "ndarray[float64]", "the new violations"),
            ("worsened", "ndarray[float64]", "the worsened ones"),
            ("df", "ndarray[float64]", "per branch: the DC flow change per MW of the project, POI to sink"),
            ("transfer_limit_mw", "float", "the largest output with no new violation (mw_max if none up to it)"),
            ("limit_found", "bool", "whether a new violation appeared below mw_max"),
            ("steps", "ndarray[float64]", "the bisection: rows (MW, new violations)"),
            ("kinds", "ndarray[int64]", "per contingency: 1 a branch, 2 a generator"),
            ("elems", "ndarray[int64]", "per contingency: its row"),
            ("converged", "ndarray[bool]", "per row, with the project: whether the AC flow converged"),
            ("flows", "ndarray[float64]", "rows x branches with the project: |S|, MVA"),
            ("vm", "ndarray[float64]", "rows x buses with the project: Vm, per unit"),
            ("success", "bool", "whether the base case with the project converged"),
        ],
        checked="PYPOWER 5.1.21, brute force: the N-1 of the case with the project at the transfer limit has no "
                "violation that the case without it lacks, and at the next output the bisection tried it has "
                "one (case30 at bus 30, case118 at buses 38 and 87; lib/power/compare_study.py).",
        example='''import vipyr as vp
r = vp.power.interconnect("case118", 38, 200.0, mw_max=1500.0)
print(len(r.new), r.transfer_limit_mw)
assert r.check()''',
    ),
    "power.cpf": dict(
        kind="python", module="_power", impl="cpf", docs="lib/power/power.docs.toml", result="PVCurve",
        check_text="this solve converged",
        raises=[
            ("ValueError", "an unknown method, a case name that is not shipped, or a case dict without "
             "baseMVA, bus, gen and branch"),
            ("module.VivStop", "the case has no bus that can be the reference, or every generator is at a "
             "reactive limit"),
        ],
        sources=["lib/power/cpf.viv", "lib/power/api.viv", "lib/power/pf.viv"],
        tests=["lib/power/cpf.viv"],
        params=[
            ("case", "str | dict", None, "a case as vp.power.runpf takes it, or a .raw path"),
            ("poi_bus", "int", None, "the bus receiving the transfer, a bus number"),
            ("sink", "str", "'slack'", "'slack' or 'gens', as for vp.power.interconnect"),
            ("step", "float", "0.5", "the first step along the curve (arc length, per unit)"),
            ("max_step", "float", "2.0", "the largest step"),
            ("stop_frac", "float", "0.7", "stop once the transfer falls below this fraction of the nose's"),
            ("max_steps", "int", "300", "the most steps"),
            ("tol", "float", "1e-10", "the corrector's tolerance, per unit"),
            ("nose_tol", "float", "0.0001", "past the nose the trace steps back and shrinks its step until "
             "the step is below this; the nose is then located to about its square"),
            ("singular_values", "bool", "False", "also compute the smallest singular value of the power-flow "
             "Jacobian at each point (dense SVD)"),
        ],
        results=[
            ("mw", "ndarray[float64]", "the transfer at each point, MW"),
            ("v_poi", "ndarray[float64]", "Vm at the POI, per unit"),
            ("v_min", "ndarray[float64]", "the lowest Vm in the network"),
            ("vm", "ndarray[float64]", "points x buses: Vm"),
            ("va", "ndarray[float64]", "points x buses: Va, radians"),
            ("nose_mw", "float", "the largest transfer on the curve, MW"),
            ("nose", "int", "the point of the nose"),
            ("smin", "ndarray[float64]", "the smallest singular value of the Jacobian per point (0 if not asked)"),
            ("success", "bool", "whether the trace passed a nose"),
        ],
        checked="PYPOWER 5.1.21's runcpf (pseudo-arclength, full trace) and the largest transfer at which "
                "PYPOWER's Newton power flow converges, on case9, case30 and case118; every curve point re-solved "
                "by PYPOWER at its transfer agrees in Vm; the Jacobian's smallest singular value at the nose is a "
                "small fraction of its base-case value (lib/power/compare_study.py).",
        example='''import vipyr as vp
r = vp.power.cpf("case118", 38, step=0.2, max_step=1.0)
print(round(r.nose_mw, 1), r.v_poi[r.nose])
assert r.check()''',
    ),
    "power.short_circuit": dict(
        kind="python", module="_power", impl="short_circuit", docs="lib/power/power.docs.toml",
        check_text="this solve converged",
        raises=[
            ("ValueError", "an unknown method, a case name that is not shipped, or a case dict without "
             "baseMVA, bus, gen and branch"),
            ("module.VivStop", "the case has no bus that can be the reference, or every generator is at a "
             "reactive limit"),
        ],
        result="ShortCircuit",
        sources=["lib/power/shortcircuit.viv", "lib/power/api.viv"],
        tests=["lib/power/shortcircuit.viv"],
        params=[
            ("case", "str | dict", None, "a case as vp.power.runpf takes it, or a .raw path (whose generators "
             "carry their source impedance ZR + jZX)"),
            ("buses", "int | list", "None", "the faulted buses (bus numbers); None for every bus"),
            ("xdss", "float", "0.2", "X''d on the machine base for a case without machine data"),
            ("cos_phi", "float", "0.85", "the generators' rated power factor, for K_G"),
        ],
        results=[
            ("buses", "ndarray[float64]", "the faulted bus numbers"),
            ("ikss_ka", "ndarray[float64]", "the initial symmetrical short-circuit current Ik'', kA"),
            ("skss_mva", "ndarray[float64]", "the short-circuit power sqrt(3) Un Ik'', MVA"),
            ("rk", "ndarray[float64]", "the Thevenin resistance at the bus, per unit on baseMVA"),
            ("xk", "ndarray[float64]", "the Thevenin reactance"),
            ("success", "bool", "always True"),
        ],
        checked="pandapower 3.5.6's shortcircuit.calc_sc (IEC 60909, three-phase, case max) on the same network "
                "(from_ppc, every generator a machine with the RAW source impedance, transformers on the system "
                "base): Ik'' at every bus of case118 (from RAW) and of ANDES's IEEE 14 and IEEE 39 RAW files "
                "(lib/power/compare_study.py, lib/power/COMPARISON.md).",
        example='''import vipyr as vp
from vipyr import _spec
r = vp.power.short_circuit(_spec.source("lib/power/cases/raw/case118_v33.raw"), buses=[38, 65])
print(r.ikss_ka)
assert r.check()''',
    ),
    # ----------------------------------------------------------------- ode
    "ode.solve": dict(
        kind="python", impl="solve", docs="lib/ode/ode.docs.toml", map="lib/ode/solve.viv:ode_solve", result="OdeSolution",
        sources=["lib/ode/solve.viv", "lib/ode/dopri5.viv", "lib/ode/radau.viv", "lib/ode/bdf.viv",
                 "lib/ode/core.viv", "lib/ode/lu.viv"],
        tests=["lib/ode/stiff_reference.viv", "lib/ode/nonstiff_events.viv", "lib/ode/lu.viv"],
        params=[
            ("rhs", "str | function", None, "the right-hand side f(t, y, p): a problem of lib/ode/problems.viv "
             "('robertson'), 'file.viv:def', a def of a module from vp.load, or a Python function in py2viv's subset"),
            ("t_span", "(float, float)", None, "the interval (t0, t1); t1 < t0 integrates backward"),
            ("y0", "ndarray[float64]", None, "the initial state, length n"),
            ("method", "str", "'auto'", "'dopri5' (or 'rk45'), 'radau', 'bdf', or 'auto': Dormand-Prince until "
             "its stiffness test fires, then Radau IIA"),
            ("rtol", "float", "0.001", "relative tolerance; raised to 100 eps if smaller"),
            ("atol", "float | ndarray[float64]", "1e-06", "absolute tolerance, one number or one per component"),
            ("params", "ndarray[float64]", "None", "the parameter vector p passed to rhs, events and jac"),
            ("t_eval", "ndarray[float64]", "None", "times to report the solution at (dense output); None reports "
             "every step"),
            ("events", "str | function", "None", "a def of the rhs's file returning one g value per event; an "
             "event is a zero of g"),
            ("direction", "ndarray[float64]", "None", "per event: +1 rising zeros only, -1 falling only, 0 both"),
            ("terminal", "ndarray[bool]", "None", "per event: stop the integration at its first zero"),
            ("jac", "str", "None", "None: exact, by reverse-mode grad of the rhs; 'fd': forward differences; "
             "or the name of a def of the rhs's file returning the n by n Jacobian"),
            ("first_step", "float", "None", "the first step; None chooses it (Hairer's rule)"),
            ("max_step", "float", "None", "the largest step allowed; None for no limit"),
            ("max_steps", "int", "1000000", "stop with status -2 after this many accepted steps"),
            ("fast", "bool", "False", "compile with the fast flags (fused multiply-add): faster, last bits may differ"),
        ],
        results=[
            ("t", "ndarray[float64]", "the output times"),
            ("y", "ndarray[float64]", "the solution, shape (n, len(t)) as in SciPy: y[i] is component i"),
            ("t_events", "list[ndarray]", "per event, the times it fired (None without events)"),
            ("y_events", "list[ndarray]", "per event, the states where it fired"),
            ("status", "int", "1: reached t1; 2: a terminal event; -1: step size underflow; -2: max_steps; "
             "-3: the solution stopped being finite"),
            ("message", "str", "the status in words"),
            ("success", "bool", "status > 0"),
            ("stats", "dict", "nfev (rhs evaluations), njev (Jacobians), nlu (LU factorizations), steps "
             "(accepted), rejected"),
            ("stiff_at", "float", "with method='auto', the time it changed to Radau IIA (None if it did not)"),
        ],
        checked="The stiff test set's published reference solutions (Mazzia and Magherini, Bari, 2008): Robertson "
                "to 1e11, Van der Pol (mu = 1000), HIRES and the Oregonator, by Radau IIA (largest relative error "
                "5.2e-10 at rtol 1e-8) and BDF (1.6e-6, as SciPy's BDF); Kepler, Lotka-Volterra, events, dense "
                "output and a batch against closed forms (lib/ode/stiff_reference.viv, lib/ode/nonstiff_events.viv); "
                "step counts against SciPy's solve_ivp (python/tests/test_ode.py).",
        example='''import vipyr as vp
# Robertson's kinetics, stiff, to t = 1e5 by Radau IIA with the exact Jacobian
r = vp.ode.solve("robertson", (0.0, 1e5), [1.0, 0.0, 0.0], method="radau", rtol=1e-8, atol=1e-12,
                 params=[0.04, 1e4, 3e7], t_eval=[1.0, 1e3, 1e5])
print(r.y[:, -1], r.stats["steps"])
assert r.check()''',
    ),
    "ode.batch": dict(
        kind="python", impl="batch", docs="lib/ode/ode.docs.toml", map="lib/ode/solve.viv:ode_batch", result="OdeBatch",
        sources=["lib/ode/solve.viv", "lib/ode/dopri5.viv", "lib/ode/radau.viv", "lib/ode/bdf.viv",
                 "lib/ode/core.viv", "lib/ode/lu.viv"],
        tests=["lib/ode/nonstiff_events.viv"],
        params=[
            ("rhs", "str | function", None, "the right-hand side f(t, y, p), as for vp.ode.solve"),
            ("t_span", "(float, float)", None, "the interval (t0, t1), the same for every solve"),
            ("y0", "ndarray[float64]", None, "the initial state, length n, or one row per solve (B, n)"),
            ("params", "ndarray[float64]", None, "one parameter vector per solve, shape (B, k)"),
            ("t_eval", "ndarray[float64]", None, "the output times, the same for every solve"),
            ("method", "str", "'auto'", "as for vp.ode.solve"),
            ("rtol", "float", "0.001", "relative tolerance"),
            ("atol", "float | ndarray[float64]", "1e-06", "absolute tolerance, one number or one per component"),
            ("events", "str | function", "None", "as for vp.ode.solve; each solve reports its first event"),
            ("direction", "ndarray[float64]", "None", "as for vp.ode.solve"),
            ("terminal", "ndarray[bool]", "None", "as for vp.ode.solve; after a terminal event the outputs are NaN"),
            ("jac", "str", "None", "as for vp.ode.solve"),
            ("first_step", "float", "None", "as for vp.ode.solve"),
            ("max_step", "float", "None", "as for vp.ode.solve"),
            ("max_steps", "int", "1000000", "as for vp.ode.solve, per solve"),
            ("fast", "bool", "False", "as for vp.ode.solve"),
        ],
        results=[
            ("t", "ndarray[float64]", "the output times (t_eval)"),
            ("y", "ndarray[float64]", "the solutions, shape (B, n, len(t))"),
            ("status", "ndarray[int64]", "each solve's status, as for vp.ode.solve"),
            ("success", "bool", "every solve reached t1 or a terminal event"),
            ("first_event", "ndarray[float64]", "each solve's first event time, NaN if none (None without events)"),
            ("stats", "dict", "per solve: nfev, njev, nlu, steps, rejected (arrays of length B)"),
        ],
        checked="40 one-compartment pharmacokinetic models against Bateman's closed form, worst error 2.8e-10 "
                "mg/L at rtol 1e-9 (lib/ode/nonstiff_events.viv); every row equal to vp.ode.solve's to the bit "
                "(python/tests/test_ode.py).",
        example='''import numpy as np
import vipyr as vp
# 1000 SIR epidemics, R0 from 1.2 to 4, recovery in 7 days
beta = np.linspace(1.2, 4.0, 1000) / 7.0
P = np.column_stack([beta, np.full(1000, 1.0 / 7.0)])
r = vp.ode.batch("sir", (0.0, 180.0), [0.999, 0.001, 0.0], P, t_eval=np.linspace(0, 180, 181),
                 method="dopri5", rtol=1e-8, atol=1e-10)
print(r.y.shape, round(r.y[-1, 2, -1], 4))     # everyone who was ever infected, at R0 = 4
assert r.check()''',
    ),
    # ----------------------------------------------------------------- gpt
    "gpt.train": dict(
        kind="python", module="_gpt", impl="train", docs="lib/gpt/gpt.docs.toml", map="lib/gpt/api.viv:gpt_train",
        raises=[
            ("ValueError", "sizes that do not fit (width not a multiple of heads, a text no longer than "
             "the context), characters or token ids outside the model's vocabulary"),
        ],
        result="GptTraining",
        sources=["lib/gpt/api.viv", "lib/gpt/gpt.viv", "lib/nn.viv", "lib/optim.viv", "lib/polar.viv", "lib/random.viv"],
        tests=["lib/gpt/gpt.viv", "lib/gpt/api.viv"],
        params=[
            ("text", "str | ndarray[int]", None, "the training text (a character-level vocabulary is made from "
             "it), or token ids"),
            ("config", "GptConfig", "None", "the sizes (vp.gpt.GptConfig); None: GptConfig() with vocab from the text"),
            ("steps", "int", "30", "training steps; the step size falls in a straight line to a tenth of its start"),
            ("seed", "int", "7", "the seed of the initial weights (Viv's rand(), Park-Miller MINSTD)"),
            ("fast", "bool", "False", "the fast build (-O3, fused multiply-adds, the platform BLAS for matrix "
             "products): about twice as fast at 200,000 parameters; results may differ in the last bits"),
        ],
        results=[
            ("model", "vp.gpt.Model", "the trained model: model.tok, model.pos, model.blocks[l].attn.heads[h].Wq, "
             "model.blocks[l].attn.Wo, model.blocks[l].mlp.up, model.unembed, with model.config and model.vocab"),
            ("losses", "ndarray[float64]", "the batch loss (mean next-token cross-entropy, nats) of every step"),
        ],
        checked="Against the NumPy program study/compare/ml/tiny_gpt.py (hand-written backward pass): its 30 "
                "losses, the first six bit for bit and all within 1.2e-15, its trained matrices within 3.6e-15 "
                "relative, and its sample to the character. Against a PyTorch 2.2.2 reimplementation "
                "(python/tests/gpt_reference.py: autograd, torch.optim.Adam, float64): 50 losses within 1.7e-14 "
                "relative at 6,560 and 32,768 parameters. The gradient against central differences of the loss "
                "(lib/gpt/gpt.viv, run by r.check()). python/tests/test_gpt.py.",
        example='''import vipyr as vp
r = vp.gpt.train("hello world! " * 5, steps=30, seed=7)
print(r.model.size, r.losses[0], r.losses[-1])   # the first loss is log 9: nine characters
print(r.model.blocks[0].attn.Wo.shape)
assert r.check()''',
    ),
    "gpt.generate": dict(
        kind="python", module="_gpt", impl="generate", docs="lib/gpt/gpt.docs.toml",
        raises=[
            ("ValueError", "sizes that do not fit (width not a multiple of heads, a text no longer than "
             "the context), characters or token ids outside the model's vocabulary"),
        ],
        map="lib/gpt/api.viv:gpt_generate", result="GptSample",
        sources=["lib/gpt/api.viv", "lib/gpt/gpt.viv", "lib/nn.viv", "lib/optim.viv", "lib/polar.viv", "lib/random.viv"],
        tests=["lib/gpt/api.viv"],
        params=[
            ("model", "vp.gpt.Model", None, "a model from vp.gpt.train (r.model)"),
            ("prompt", "str | ndarray[int]", None, "the text (or token ids) to continue"),
            ("n", "int", "100", "how many tokens to add"),
            ("temperature", "float", "1.0", "the logits are divided by it before the softmax; 0: the most likely "
             "token every time"),
            ("seed", "int", "11", "the seed of the draws"),
        ],
        results=[
            ("tokens", "ndarray[int64]", "the prompt's ids followed by the new ones"),
            ("text", "str", "the same as characters (None for a model trained on token ids)"),
        ],
        checked="The sample of study/compare/ml/tiny_gpt.py from the same model and seed, to the character; at "
                "temperature 1 the draws of lib/gpt's sample (lib/gpt/api.viv). python/tests/test_gpt.py.",
        example='''import vipyr as vp
m = vp.gpt.train("hello world! " * 5, steps=30, seed=7).model
r = vp.gpt.generate(m, "hello", n=24, seed=11)
print(r.text)
assert r.check()''',
    ),
    "gpt.attention": dict(
        kind="python", module="_gpt", impl="attention", docs="lib/gpt/gpt.docs.toml",
        raises=[
            ("ValueError", "sizes that do not fit (width not a multiple of heads, a text no longer than "
             "the context), characters or token ids outside the model's vocabulary"),
        ],
        map="lib/gpt/api.viv:gpt_attention", result="GptAttention",
        sources=["lib/gpt/api.viv", "lib/gpt/gpt.viv", "lib/nn.viv", "lib/optim.viv", "lib/polar.viv", "lib/random.viv"],
        tests=["lib/gpt/api.viv"],
        params=[
            ("model", "vp.gpt.Model", None, "a model from vp.gpt.train (r.model)"),
            ("prompt", "str | ndarray[int]", None, "one sequence of 1 to context tokens"),
        ],
        results=[
            ("weights", "ndarray[float64]", "shape (layers, heads, T, T): row i of a head's matrix is how much "
             "position i takes from positions 0 to i (each row sums to 1)"),
            ("labels", "list[str]", "the prompt's characters, for the axes (None for a model trained on token ids)"),
        ],
        checked="Rows summing to 1 and zero above the diagonal (r.check()); the weights are those of the "
                "attention the forward pass runs (lib/nn's attention_weights, the same operations).",
        example='''import vipyr as vp
m = vp.gpt.train("hello world! " * 5, steps=30, seed=7).model
r = vp.gpt.attention(m, "hello wo")
print(r.weights.shape, r.weights[0, 0, -1].round(3))   # where the last position of head 0 looks
assert r.check()''',
    ),
    "gpt.loss": dict(
        kind="python", module="_gpt", impl="loss", docs="lib/gpt/gpt.docs.toml", map="lib/gpt/api.viv:gpt_loss",
        raises=[
            ("ValueError", "sizes that do not fit (width not a multiple of heads, a text no longer than "
             "the context), characters or token ids outside the model's vocabulary"),
        ],
        result="GptLoss",
        sources=["lib/gpt/api.viv", "lib/gpt/gpt.viv", "lib/nn.viv", "lib/optim.viv", "lib/polar.viv", "lib/random.viv"],
        tests=["lib/gpt/api.viv"],
        params=[
            ("model", "vp.gpt.Model", None, "a model from vp.gpt.train (r.model), its arrays as they are now"),
            ("tokens", "str | ndarray[int]", None, "text (or token ids) longer than the model's context"),
        ],
        results=[
            ("loss", "float", "the mean next-token cross-entropy (nats) over every window of context tokens"),
        ],
        checked="The loss of the PyTorch reimplementation (python/tests/gpt_reference.py) on the same windows, "
                "within 1e-12 relative (python/tests/test_gpt.py).",
        example='''import vipyr as vp
text = "hello world! " * 5
m = vp.gpt.train(text, steps=30, seed=7).model
print(vp.gpt.loss(m, text).loss)
m.blocks[0].mlp.up[:] = 0.0                      # the arrays are the model: edit one and measure again
r = vp.gpt.loss(m, text)
print(r.loss)
assert r.check()''',
    ),
    # ---------------------------------------------------------------- gguf
    "gguf.read": dict(
        kind="python", module="_gguf", impl="read", docs="lib/gguf/gguf.docs.toml", map="lib/gguf/api.viv:file_index",
        mixin="_gguf:GgufFileMixin",
        raises=[
            ("VivStop", "a file that is not GGUF (it does not start with the bytes `GGUF`), a version other than 3, "
             "a value type GGUF does not define, a string that is not UTF-8, or a table that runs past the end "
             "of the file"),
        ],
        result="GgufFile",
        sources=["lib/gguf/api.viv", "lib/gguf/header.viv", "lib/gguf/tensors.viv", "python/vipyr/_gguf.py"],
        tests=["lib/gguf/tests/header.viv", "lib/gguf/tests/tensors.viv"],
        params=[
            ("path", "str | Path", None, "the GGUF file"),
        ],
        results=[
            ("path", "str", "the file read"),
            ("metadata", "dict", "key -> int, float, bool, str or list (an array; an array of arrays is a list of "
             "lists), in the file's order"),
            ("tensors", "list[TensorInfo]", "the tensor table in the file's order: name, shape (NumPy's order, "
             "the reverse of GGUF's dims), type (ggml's name), offset from the file's start, n_bytes"),
            ("alignment", "int", "general.alignment, or 32 when the file does not set it"),
            ("data_offset", "int", "where the data section starts: the end of the table rounded up to the alignment"),
        ],
        checked="gguf-py 0.19.0's GGUFReader: the metadata and the tensor table of lib/gguf/tests/tiny.gguf (every "
                "value type) and of llama2.c's stories260K, stories15M-q8_0 and stories15M-q4_0, printed the same "
                "way and identical byte for byte in the interpreter, as C and as wasm (tools/gguf/check.sh); "
                "python/tests/test_gguf.py.",
        example='''import os
import vipyr as vp
r = vp.gguf.read(os.path.join(vp.viv_root(), "lib/gguf/tests/stories260K.gguf"))
print(r.metadata["general.architecture"], r.metadata["llama.block_count"], len(r.metadata["tokenizer.ggml.tokens"]))
print(r.tensor_info("blk.0.attn_q.weight"))
assert r.check()''',
    ),
    "gguf.tensor": dict(
        kind="python", module="_gguf", impl="tensor", docs="lib/gguf/gguf.docs.toml", map="lib/gguf/api.viv:dequant",
        raises=[
            ("KeyError", "no tensor of that name"),
            ("NotImplementedError", "a type lib/gguf does not read (the IQ types, Q2_K, Q3_K, Q5_0, Q5_1, Q5_K, ...)"),
        ],
        result="GgufTensor",
        sources=["lib/gguf/api.viv", "lib/gguf/quants.viv", "lib/gguf/tensors.viv", "python/vipyr/_gguf.py"],
        tests=["lib/gguf/tests/tensors.viv", "lib/gguf/tests/quants.viv"],
        params=[
            ("path", "str | Path", None, "the GGUF file"),
            ("name", "str", None, "the tensor's name (`blk.0.attn_q.weight`)"),
            ("file", "GgufFile", "None", "the result of vp.gguf.read(path), to reuse its table; None reads it"),
        ],
        results=[
            ("array", "ndarray[float32]", "the tensor's values in its shape (NumPy's order), float32 as ggml "
             "computes in; every quantized value is exact in float32"),
            ("info", "TensorInfo", "its row of the tensor table"),
        ],
        checked="gguf-py 0.19.0's gguf.quants.dequantize: every tensor of lib/gguf/tests/tiny.gguf (one of each "
                "type lib/gguf reads) and of stories260K, stories15M-q8_0 and stories15M-q4_0, number for number "
                "as binary32 bit patterns, in the interpreter, as C and as wasm (tools/gguf/check.sh); "
                "lib/gguf/tests/tiny.ref holds gguf-py's values for tiny.gguf (python/tests/test_gguf.py).",
        example='''import os
import vipyr as vp
path = os.path.join(vp.viv_root(), "lib/gguf/tests/stories260K.gguf")
r = vp.gguf.tensor(path, "blk.0.attn_q.weight")
print(r.info.type, r.array.shape, r.array.dtype, r.array[0, :3])
assert r.check()''',
    ),
    # ------------------------------------------------- optimize (python)
    "optimize.least_squares": dict(
        kind="python", module="_optimize", impl="least_squares", docs="lib/minpack/optimize.docs.toml",
        result="LeastSquaresResult", positional=["x0"],
        sources=["lib/minpack/fit.viv", "lib/minpack/lmder.viv", "lib/minpack/lmpar.viv", "lib/minpack/qrfac.viv",
                 "lib/minpack/qrsolv.viv", "lib/minpack/enorm.viv"],
        tests=["lib/minpack/fit.viv"],
        check_text="lmder stopped on a tolerance (status > 0)",
        raises=[
            ("ValueError", "a tolerance below machine epsilon, fewer residuals than variables, or an array of the "
             "wrong shape; vipyr._optimize.OptimizeProblemError (a ValueError) when the function or jac cannot be "
             "made into a Viv problem, or py2viv refuses a Python function"),
            ("vipyr.VivCompileError", "a SyntaxError: the function's .viv file (or py2viv's translation) does not "
             "compile"),
        ],
        params=[
            ("fun", "str | function", None, "the residuals r(x, *args), an array of m: 'file.viv:def', a def of a "
             "module from vp.load, or a Python function in py2viv's subset"),
            ("x0", "ndarray[float64]", None, "the starting point, length n <= m"),
            ("jac", "str | function", "None", "None: exact, by reverse-mode grad of the residuals; 'fd' (or "
             "'2-point'): MINPACK's forward differences; or a def (a name of fun's file, or a function) "
             "jac(x, *args) returning the m by n Jacobian"),
            ("args", "tuple", "()", "extra arguments of fun and jac, each a number or a 1-D array"),
            ("ftol", "float", "1e-08", "stop when the relative reduction of the sum of squares is at most ftol"),
            ("xtol", "float", "1e-08", "stop when the relative change of x is at most xtol"),
            ("gtol", "float", "1e-08", "stop when the largest cosine between the residuals and a column of the "
             "Jacobian is at most gtol"),
            ("x_scale", "float | ndarray[float64] | str", "1.0", "the scale of each variable (MINPACK's diag is "
             "1 / x_scale); 'jac' scales by the Jacobian's column norms"),
            ("max_nfev", "int", "None", "the residual evaluations allowed; None: 100 n, or 100 n (n + 1) with "
             "jac='fd', as SciPy"),
            ("diff_step", "float", "None", "with jac='fd', the relative step; None: sqrt(eps)"),
            ("fast", "bool", "False", "compile with the fast flags (fused multiply-add): faster, last bits may differ"),
        ],
        results=[
            ("x", "ndarray[float64]", "the solution"),
            ("cost", "float", "half the sum of squares of the residuals at x"),
            ("fun", "ndarray[float64]", "the residuals at x"),
            ("jac", "ndarray[float64]", "the m by n Jacobian at x"),
            ("grad", "ndarray[float64]", "the gradient of cost at x, jac^T fun"),
            ("optimality", "float", "the largest absolute entry of grad"),
            ("active_mask", "ndarray[int64]", "zeros: method 'lm' has no bounds"),
            ("nfev", "int", "residual evaluations (with jac='fd', including the n of each difference Jacobian, as "
             "SciPy counts them)"),
            ("njev", "int", "Jacobian evaluations"),
            ("status", "int", "SciPy's: -1 improper input, 0 max_nfev reached, 1 gtol, 2 ftol, 3 xtol, 4 ftol and "
             "xtol"),
            ("message", "str", "the status in words"),
            ("success", "bool", "status > 0"),
            ("info", "int", "MINPACK's info, 0 to 8, which status is made from"),
        ],
        checked="lib/minpack/lmder.viv matches a gfortran build of netlib's lmder on 24 Moré-Garbow-Hillstrom "
                "cases (lib/minpack/mgh_check.viv); through vp.optimize, NIST's 27 StRD nonlinear regression "
                "problems from both certified starts and 18 MGH cases against SciPy 1.12's least_squares(method="
                "'lm') (lib/minpack/bench_optimize.py, lib/minpack/bench_optimize.json; python/tests/test_optimize.py).",
        example='''import numpy as np
import vipyr as vp

def rosenbrock(x):
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])

# translated to Viv by py2viv once, compiled with lmder, Jacobian by grad
r = vp.optimize.least_squares(rosenbrock, [-1.2, 1.0])
print(r.x, r.nfev, r.njev, r.status)
assert r.check()''',
    ),
    "optimize.curve_fit": dict(
        kind="python", module="_optimize", impl="curve_fit", docs="lib/minpack/optimize.docs.toml",
        result="CurveFitResult", positional=["p0"], unpack=2,
        sources=["lib/minpack/fit.viv", "lib/minpack/lmder.viv", "lib/minpack/lmpar.viv", "lib/minpack/qrfac.viv",
                 "lib/minpack/qrsolv.viv", "lib/minpack/enorm.viv"],
        tests=["lib/minpack/fit.viv"],
        check_text="lmder stopped on a tolerance (info 1 to 4)",
        raises=[
            ("ValueError", "a tolerance below machine epsilon, fewer residuals than variables, or an array of the "
             "wrong shape; vipyr._optimize.OptimizeProblemError (a ValueError) when the function or jac cannot be "
             "made into a Viv problem, or py2viv refuses a Python function"),
            ("vipyr.VivCompileError", "a SyntaxError: the function's .viv file (or py2viv's translation) does not "
             "compile"),
        ] + [("RuntimeError", "lmder stopped without converging (info 0 or 5 to 8), as SciPy's "
                          "'Optimal parameters not found'")],
        params=[
            ("f", "str | function", None, "the model at one point, f(x, b1, ..., bk) or f(x, p) with p an "
             "array: 'file.viv:def', a def of a module from vp.load, or a Python function in py2viv's subset"),
            ("xdata", "ndarray[float64]", None, "the independent variable, one number per point (length m)"),
            ("ydata", "ndarray[float64]", None, "the observations, length m"),
            ("p0", "ndarray[float64]", "None", "the starting parameters; None: ones, one per parameter of f"),
            ("sigma", "ndarray[float64]", "None", "the standard deviation of each observation; the fit minimizes "
             "sum(((f(x) - y) / sigma)^2)"),
            ("absolute_sigma", "bool", "False", "True: pcov is (J^T J)^-1 with sigma taken as absolute; False: "
             "scaled by the reduced chi-square rss / (m - n), as SciPy"),
            ("jac", "str", "None", "None: exact, reverse-mode grad of the model at each point; 'fd': MINPACK's "
             "forward differences (what SciPy's curve_fit uses)"),
            ("ftol", "float", "1.49012e-08", "leastsq's ftol"),
            ("xtol", "float", "1.49012e-08", "leastsq's xtol"),
            ("gtol", "float", "0.0", "leastsq's gtol"),
            ("maxfev", "int", "None", "the residual evaluations allowed; None: 100 (n + 1), or 200 (n + 1) with "
             "jac='fd', as leastsq"),
            ("epsfcn", "float", "None", "with jac='fd', the relative step squared; None: eps"),
            ("diag", "ndarray[float64]", "None", "MINPACK's variable scales; None: the Jacobian's column norms"),
            ("fast", "bool", "False", "compile with the fast flags (fused multiply-add): faster, last bits may differ"),
        ],
        results=[
            ("popt", "ndarray[float64]", "the fitted parameters"),
            ("pcov", "ndarray[float64]", "their covariance, n by n (infinite if the Jacobian is rank deficient)"),
            ("perr", "ndarray[float64]", "the standard errors, sqrt(diag(pcov))"),
            ("fun", "ndarray[float64]", "the weighted residuals (f(x) - y) / sigma at popt"),
            ("rss", "float", "their sum of squares"),
            ("nfev", "int", "model evaluations over all points, counted as leastsq counts them"),
            ("njev", "int", "Jacobian evaluations"),
            ("info", "int", "MINPACK's info, 1 to 4"),
            ("message", "str", "info in words"),
            ("success", "bool", "always True: a fit that does not converge raises RuntimeError"),
        ],
        checked="NIST's 26 single-predictor StRD nonlinear regression problems from both certified starts: the "
                "certified parameters and standard deviations, and SciPy 1.12's curve_fit counts "
                "(lib/minpack/bench_optimize.py, lib/minpack/bench_optimize.json); Misra1a in lib/minpack/fit.viv.",
        example='''import numpy as np
import vipyr as vp

# NIST StRD Misra1a, from its first certified start
x = [77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8, 378.4, 434.8, 477.3, 536.8, 593.1, 689.1, 760.0]
y = [10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02, 44.82, 50.76, 55.05, 61.01, 66.40, 75.47, 81.78]

def misra1a(x, b1, b2):
    return b1 * (1.0 - np.exp(-b2 * x))

popt, pcov = vp.optimize.curve_fit(misra1a, x, y, [500.0, 1e-4])
print(popt, np.sqrt(np.diag(pcov)))      # NIST: 238.94212918, 5.5015643181e-4; sd 2.7070075241, 7.2668688436e-6
r = vp.optimize.curve_fit(misra1a, x, y, [500.0, 1e-4])
assert r.check()''',
    ),
    "optimize.root": dict(
        kind="python", module="_optimize", impl="root", docs="lib/minpack/optimize.docs.toml",
        result="RootResult", positional=["x0"],
        sources=["lib/minpack/fit.viv", "lib/minpack/lmder.viv", "lib/minpack/lmpar.viv", "lib/minpack/qrfac.viv",
                 "lib/minpack/qrsolv.viv", "lib/minpack/enorm.viv"],
        tests=["lib/minpack/fit.viv"],
        check_text="lmder stopped on a tolerance (status 1 to 4)",
        raises=[
            ("ValueError", "a tolerance below machine epsilon, fewer residuals than variables, or an array of the "
             "wrong shape; vipyr._optimize.OptimizeProblemError (a ValueError) when the function or jac cannot be "
             "made into a Viv problem, or py2viv refuses a Python function"),
            ("vipyr.VivCompileError", "a SyntaxError: the function's .viv file (or py2viv's translation) does not "
             "compile"),
        ],
        params=[
            ("fun", "str | function", None, "the system F(x, *args) = 0, n equations: 'file.viv:def', a def of a "
             "module from vp.load, or a Python function in py2viv's subset"),
            ("x0", "ndarray[float64]", None, "the starting point, length n"),
            ("args", "tuple", "()", "extra arguments of fun and jac, each a number or a 1-D array"),
            ("jac", "str | function", "None", "as for vp.optimize.least_squares"),
            ("method", "str", "'lm'", "'lm' only: Levenberg-Marquardt on the square system, as SciPy's "
             "root(method='lm'); MINPACK's hybrd/hybrj (Powell's hybrid method, SciPy's default 'hybr') is not ported"),
            ("xtol", "float", "1.49012e-08", "leastsq's xtol"),
            ("ftol", "float", "1.49012e-08", "leastsq's ftol"),
            ("gtol", "float", "0.0", "leastsq's gtol"),
            ("maxfev", "int", "None", "as for vp.optimize.curve_fit"),
            ("eps", "float", "None", "with jac='fd', the relative step squared; None: eps"),
            ("diag", "ndarray[float64]", "None", "MINPACK's variable scales; None: the Jacobian's column norms"),
            ("fast", "bool", "False", "compile with the fast flags (fused multiply-add): faster, last bits may differ"),
        ],
        results=[
            ("x", "ndarray[float64]", "the root"),
            ("fun", "ndarray[float64]", "F at x"),
            ("jac", "ndarray[float64]", "the n by n Jacobian at x"),
            ("success", "bool", "status is 1 to 4"),
            ("status", "int", "MINPACK's info, 0 to 8"),
            ("message", "str", "status in words"),
            ("nfev", "int", "evaluations of F"),
            ("njev", "int", "Jacobian evaluations"),
        ],
        checked="SciPy 1.12's root(method='lm') on the square Moré-Garbow-Hillstrom problems (Rosenbrock, "
                "helical valley, Powell singular, Freudenstein-Roth) from their standard starts: same root and "
                "evaluation counts (lib/minpack/bench_optimize.py, python/tests/test_optimize.py).",
        example='''import numpy as np
import vipyr as vp

def helical_valley(x):
    th = np.arctan(x[1] / x[0]) / (2.0 * np.pi)
    if x[0] < 0.0:
        th = th + 0.5
    return np.array([10.0 * (x[2] - 10.0 * th), 10.0 * (np.sqrt(x[0] ** 2 + x[1] ** 2) - 1.0), x[2]])

r = vp.optimize.root(helical_valley, [-1.0, 0.0, 0.0])
print(r.x, r.nfev, r.njev)
assert r.check()''',
    ),
    # ------------------------------------------- inferred (no manifest yet)
    "finance.bs_price": dict(
        kind="inferred", source="examples/options/bs.viv", entry="bs_price",
        results=[("price", "Num", "the option's value, in the currency of S and K")],
        checked="The SciPy closed form on 2000 random quotes: worst relative error 1.0e-13, tolerance 1e-12 "
                "(examples/options/test_options.py).",
        example='''import vipyr as vp
r = vp.finance.bs_price(S=42.0, K=40.0, T=0.5, r=0.1, q=0.0, sigma=0.2, is_call=True)
print(r.price)                            # Hull's example: 4.76
assert r.check()''',
    ),
    "finance.implied_vol": dict(
        kind="inferred", source="examples/options/bs.viv", entry="implied_vol",
        results=[("sigma", "Num", "the volatility that reproduces the price, per unit sqrt(year); NaN outside the no-arbitrage bounds")],
        checked="A round trip through bs_price on a 1080-quote grid: worst relative error 8.8e-13 on the 898 quotes "
                "with time value above 1e-5 of the price, tolerance 1e-12 (examples/options/test_options.py).",
        example='''import vipyr as vp
r = vp.finance.implied_vol(price=4.7594, S=42.0, K=40.0, T=0.5, r=0.1, q=0.0, is_call=True)
print(r.sigma)                            # 0.2
assert r.check()''',
    ),
    "finance.crr_price": dict(
        kind="inferred", source="examples/options/crr.viv", entry="crr_price",
        results=[("price", "Num", "the option's value on an N-step Cox-Ross-Rubinstein tree")],
        checked="Hull's printed American put values 4.49, 4.263, 4.272, 4.278 and 4.283 (N = 5 to 500), within "
                "half a unit of the last printed digit; the European value at N = 4000 within 2e-3 of Black-Scholes.",
        example='''import vipyr as vp
r = vp.finance.crr_price(S=50.0, K=50.0, T=5 / 12, r=0.1, q=0.0, sigma=0.4, N=5,
                            is_call=False, american=True)
print(r.price)                            # Hull: 4.49
assert r.check()''',
    ),
    "finance.heston_cos": dict(
        kind="inferred", source="examples/options/heston.viv", entry="heston_cos",
        results=[("prices", "Arr", "one price per strike")],
        checked="Fang and Oosterlee (2008): 5.785155450 at T = 1 (error 1.6e-8, tolerance 2e-8); SciPy quad of the "
                "Gil-Pelaez integrals on 21 options, within 2.0e-13 (tolerance 1e-9).",
        example='''import numpy as np
import vipyr as vp
r = vp.finance.heston_cos(S=100.0, K=np.array([100.0]), T=1.0, r=0.0, q=0.0, v0=0.0175,
                             kappa=1.5768, theta=0.0398, eta=0.5751, rho=-0.5711,
                             N=1024, L=32.0, is_call=np.array([1.0]))
print(r.prices)                           # Fang-Oosterlee: 5.785155450
assert r.check()                          # the program's own test, at this N and L''',
    ),
    "finance.cn_american_put": dict(
        kind="inferred", source="examples/options/cnpde.viv", entry="cn_american_put",
        results=[("price", "Num", "the American put's value at S0")],
        checked="The CRR tree, the mean of N = 10000 and 10001 (4.48669): error 8.8e-5, tolerance 2e-4.",
        example='''import vipyr as vp
r = vp.finance.cn_american_put(S0=50.0, K=50.0, T=5 / 12, r=0.1, q=0.0, sigma=0.4,
                                  Smax=200.0, M=400, Nt=200, implicit=4)
print(r.price)                            # the tree gives 4.48669
assert r.check()''',
    ),
    "finance.bs_greeks": dict(
        kind="inferred", source="examples/options/greeks.viv", entry="bs_greeks",
        results=[("delta", "Num", "dV/dS"), ("gamma", "Num", "d2V/dS2, per unit of S"),
                 ("vega", "Num", "dV/dsigma")],
        checked="The closed-form Black-Scholes Greeks (examples/options/test_options.py, "
                "test_bs_greeks_by_autodiff); delta and vega come from Viv's reverse-mode `grad`.",
        example='''import vipyr as vp
r = vp.finance.bs_greeks(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.2, is_call=True)
print(r.delta, r.gamma, r.vega)
assert r.check()''',
    ),
    "randla.trace_hutchpp": dict(
        kind="inferred", source="lib/randla/api.viv", entry="trace_hutchpp",
        results=[("trace", "Num", "the estimate of the trace of A")],
        checked="lib/randla/check_ref.py: on a 400 by 400 matrix with eigenvalues j^-2, the mean relative "
                "error over 8 seeds falls with the number of products m with a fitted slope above 0.9 "
                "(Hutchinson's falls as m^-1/2).",
        example='''import numpy as np
import vipyr as vp
rng = np.random.default_rng(0)
Q, _ = np.linalg.qr(rng.standard_normal((200, 200)))
lam = np.arange(1, 201) ** -2.0
r = vp.randla.trace_hutchpp(A=(Q * lam) @ Q.T, m=48, seed=0)
print(r.trace, lam.sum())                 # the estimate and the exact trace
assert r.check()''',
    ),
    "randla.rp_cholesky": dict(
        kind="inferred", source="lib/randla/api.viv", entry="rp_cholesky",
        results=[("F", "Arr", "the n by k factor, with A approximately F F^T")],
        checked="lib/randla/check_ref.py: on a Gaussian kernel of 500 points (bandwidth 0.05), the trace error "
                "tr(A - F F^T) at k = 40 is below the best rank-20 approximation's.",
        example='''import numpy as np
import vipyr as vp
x = np.linspace(0.0, 1.0, 300)
K = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / 0.05**2)
r = vp.randla.rp_cholesky(A=K, k=40, seed=7)
print(np.trace(K) - np.sum(r.F * r.F))   # the trace error
assert r.check()''',
    ),
    # ------------------------------------------------------------ features
    **{
        "features." + n: dict(
            kind="feature", entry=n, source="catalog/%s.viv" % src,
            example='''import vipyr as vp
r = vp.features.%s()                %s# a synthetic table; pass df= for yours
print(r.frame.head())
assert r.check()''' % (n, " " * (10 - len(n))),
        )
        for n, src in [("sessions", "sessions"), ("velocity", "velocity"), ("streaks", "streaks"),
                       ("ewma", "ewma"), ("since_last", "since_last"), ("delinquency", "delinquency")]
    },
    # ------------------------------------------------- stats (python)
    "stats.describe": dict(
        kind="python", module="_stats", impl="describe", docs="lib/stats/stats.docs.toml",
        result="DescribeResult", positional=["x"],
        sources=["lib/stats/describe.viv", "lib/stats/api.viv"], tests=["lib/stats/describe.viv"],
        raises=[("ValueError", "fewer than two values, or a method that is not registered")],
        params=[
            ("x", "ndarray[float64]", None, "the sample, any shape (flattened)"),
            ("ddof", "int", "1", "the variance divides by n - ddof"),
            ("method", "str", "'two_pass'", "the variance slot: 'two_pass' (corrected two-pass, pairwise sums) "
             "or 'welford' (one pass)"),
        ],
        results=[
            ("nobs", "int", "the number of values"),
            ("minmax", "tuple", "the smallest and largest value"),
            ("mean", "float", "the mean"),
            ("variance", "float", "the variance, divided by n - ddof"),
            ("skewness", "float", "g1, the biased sample skewness (SciPy's skew default)"),
            ("kurtosis", "float", "g2, the biased excess kurtosis (SciPy's kurtosis default)"),
        ],
        checked="lib/stats/describe.viv's tests; python/tests/test_stats.py against NumPy and SciPy, and on "
                "1e9 + x samples where the textbook formula sum x^2 - n mean^2 loses every digit.",
        example='''import numpy as np
import vipyr as vp
x = 1e9 + np.array([4.0, 7.0, 13.0, 16.0])
r = vp.stats.describe(x)
print(r.mean, r.variance)                  # 1000000010.0 30.0
assert r.check()''',
    ),
    "stats.quantile": dict(
        kind="python", module="_stats", impl="quantile", docs="lib/stats/stats.docs.toml",
        result="QuantileResult", positional=["x", "q"],
        sources=["lib/stats/describe.viv", "lib/stats/api.viv"], tests=["lib/stats/describe.viv"],
        raises=[("ValueError", "a type outside 1 to 9, or q outside [0, 1]")],
        params=[
            ("x", "ndarray[float64]", None, "the sample"),
            ("q", "float | ndarray[float64]", None, "the probabilities"),
            ("type", "int", "7", "Hyndman and Fan's type, 1 to 9 (R's quantile(type = ); 7 is R's and NumPy's "
             "default)"),
        ],
        results=[("values", "float | ndarray[float64]", "the sample quantiles, q's shape"),
                 ("type", "int", "the type used")],
        checked="R 4.3.3's quantile() for all nine types (lib/stats/describe.viv, python/tests/test_stats.py).",
        example='''import vipyr as vp
x = [2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0, 8.0]
for t in (1, 6, 7, 8):
    print(t, vp.stats.quantile(x, [0.25, 0.5], type=t).values)
r = vp.stats.quantile(x, 0.9)
assert r.check()''',
    ),
    "stats.ols": dict(
        kind="python", module="_stats", impl="ols", docs="lib/stats/stats.docs.toml", mixin="_stats:OLSMixin",
        result="OLSResult", positional=["y", "X"],
        sources=["lib/stats/linmod.viv", "lib/stats/api.viv", "lib/stats/special.viv", "lib/stats/dist.viv"],
        tests=["lib/stats/linmod.viv"],
        raises=[("ValueError", "a solver that is not registered, or y and X of different lengths")],
        params=[
            ("y", "ndarray[float64]", None, "the response, n values"),
            ("X", "ndarray[float64]", None, "the n by k design matrix; a constant column is the intercept "
             "(add one yourself)"),
            ("weights", "ndarray[float64]", "None", "weights w_i: weighted least squares, minimizing "
             "sum w_i r_i^2"),
            ("solver", "str", "'qr'", "the least-squares slot: 'qr' (pivoted Householder QR), 'cholesky' "
             "(normal equations) or 'svd' (pseudo-inverse)"),
            ("refine", "bool", "True", "with 'qr', one step of iterative refinement with an exactly summed "
             "residual"),
            ("tol", "float", "1e-10", "rank: a pivot below tol times the first (QR), or a singular value below "
             "tol times the largest (SVD), drops its column"),
            ("names", "list", "None", "names of the columns (a DataFrame's columns by default)"),
        ],
        results=[
            ("coef", "ndarray[float64]", "the coefficients (NaN for a dropped column)"),
            ("bse", "ndarray[float64]", "their standard errors"),
            ("tvalues", "ndarray[float64]", "coef / bse"),
            ("pvalues", "ndarray[float64]", "two-sided p-values from Student's t with df_resid"),
            ("conf_int", "ndarray[float64]", "95% confidence intervals, k by 2"),
            ("rsquared", "float", "R^2 (centered when there is a constant column, as statsmodels)"),
            ("rsquared_adj", "float", "adjusted R^2"),
            ("fvalue", "float", "the F statistic of the model against the constant"),
            ("f_pvalue", "float", "its p-value"),
            ("ssr", "float", "the residual sum of squares (weighted)"),
            ("centered_tss", "float", "the total sum of squares about the (weighted) mean"),
            ("scale", "float", "the residual variance ssr / df_resid"),
            ("resid_sd", "float", "the residual standard deviation, sqrt(scale)"),
            ("df_model", "int", "rank minus the constant"),
            ("df_resid", "int", "n minus rank"),
            ("nobs", "int", "n"),
            ("rank", "int", "the rank found by the solver"),
            ("llf", "float", "the Gaussian log-likelihood"),
            ("aic", "float", "Akaike's criterion, statsmodels' convention (R's AIC counts sigma: + 2)"),
            ("bic", "float", "the Bayesian criterion, statsmodels' convention"),
            ("resid", "ndarray[float64]", "y - X coef"),
            ("fittedvalues", "ndarray[float64]", "X coef"),
            ("cov_params", "ndarray[float64]", "scale (X^T W X)^-1"),
            ("names", "list", "the column names"),
            ("solver", "str", "the solver used"),
            ("weighted", "bool", "whether weights were given"),
            ("k_constant", "int", "1 when X has a constant column"),
        ],
        checked="NIST StRD's 11 linear regression datasets (Norris, Pontius, NoInt1, NoInt2, Filip, Longley, "
                "Wampler1-5): certified digits of every coefficient, standard error, the residual SD and R^2, "
                "beside statsmodels 0.14 and R 4.3.3's lm (lib/stats/nist_lls.py, lib/stats/nist_lls.json).",
        example='''import numpy as np
import vipyr as vp
rng = np.random.default_rng(1)
x = rng.uniform(0, 10, 50)
X = np.column_stack([np.ones(50), x])
y = 2.0 + 0.5 * x + rng.normal(0, 1, 50)
r = vp.stats.ols(y, X, names=["const", "x"])
print(r.summary())
assert r.check()''',
    ),
    "stats.glm": dict(
        kind="python", module="_stats", impl="glm", docs="lib/stats/stats.docs.toml", mixin="_stats:GLMMixin",
        result="GLMResult", positional=["y", "X"],
        sources=["lib/stats/glm.viv", "lib/stats/linmod.viv", "lib/stats/api.viv", "lib/stats/special.viv",
                 "lib/stats/dist.viv"],
        tests=["lib/stats/glm.viv"],
        raises=[("ValueError", "a family, link, fitting method or cov that is not one of the registered"),
                ("vipyr._stats.GLMWarning", "a warning, with R's text: the iteration did not converge, or fitted "
                 "probabilities or rates numerically 0 or 1 (separation)")],
        params=[
            ("y", "ndarray[float64]", None, "the response: 0/1 or proportions (binomial), counts (poisson), "
             "positive values (gamma)"),
            ("X", "ndarray[float64]", None, "the n by k design matrix"),
            ("family", "str", "'binomial'", "'binomial', 'poisson', 'gamma' or 'gaussian'"),
            ("link", "str", "None", "the link: logit or probit (binomial), log (poisson), inverse or log "
             "(gamma), identity (gaussian); None: the canonical one"),
            ("weights", "ndarray[float64]", "None", "prior weights (binomial: the numbers of trials)"),
            ("offset", "ndarray[float64]", "None", "a known term added to the linear predictor"),
            ("fit", "str", "'irls'", "the fitting slot: 'irls' (R's glm.fit, Fisher scoring) or 'newton' "
             "(Newton's method with the observed information)"),
            ("cov", "str", "'expected'", "'expected': R's (X^T W X)^-1; 'observed': the inverse of the "
             "observed information, its Hessian derivative by deriv (differs for probit and gamma-log)"),
            ("epsilon", "float", "1e-08", "R's convergence tolerance on the relative change of the deviance"),
            ("maxit", "int", "25", "the iterations allowed (R's default)"),
            ("names", "list", "None", "names of the columns"),
        ],
        results=[
            ("coef", "ndarray[float64]", "the coefficients"),
            ("bse", "ndarray[float64]", "their standard errors"),
            ("tvalues", "ndarray[float64]", "coef / bse (z for binomial and poisson, t for gamma and gaussian)"),
            ("pvalues", "ndarray[float64]", "two-sided p-values, as R's summary.glm"),
            ("conf_int", "ndarray[float64]", "95% Wald intervals"),
            ("deviance", "float", "the residual deviance"),
            ("null_deviance", "float", "the deviance of the intercept-only model"),
            ("df_resid", "int", "n minus rank"),
            ("df_null", "int", "n minus the intercept"),
            ("dispersion", "float", "1 (binomial, poisson) or the Pearson chi-squared over df_resid"),
            ("aic", "float", "R's AIC"),
            ("iterations", "int", "the iterations taken"),
            ("converged", "bool", "whether R's rule stopped it"),
            ("mu", "ndarray[float64]", "the fitted means"),
            ("eta", "ndarray[float64]", "the linear predictor"),
            ("resid_deviance", "ndarray[float64]", "the deviance residuals"),
            ("resid_pearson", "ndarray[float64]", "the Pearson residuals"),
            ("cov_params", "ndarray[float64]", "dispersion times the unscaled covariance"),
            ("warnings", "tuple", "R's warnings, as text"),
            ("family", "str", "the family"),
            ("link", "str", "the link"),
            ("fit", "str", "the fitting method"),
            ("cov", "str", "which information the covariance inverts"),
            ("names", "list", "the column names"),
            ("nobs", "int", "n"),
            ("rank", "int", "the rank of X"),
        ],
        checked="R 4.3.3's glm on its own datasets (esoph-style binomial with trials, mtcars logit and probit, "
                "warpbreaks Poisson, a Gamma fit with both links) and a separated logistic fit, to 1e-10 at "
                "R's epsilon 1e-14 (lib/stats/glm_check.py, python/tests/test_stats.py).",
        example='''import numpy as np
import vipyr as vp
x = np.arange(1.0, 11.0)
y = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1.0])
X = np.column_stack([np.ones(10), x])
r = vp.stats.glm(y, X, family="binomial", link="probit", names=["const", "x"])
print(r.summary())
assert r.check()''',
    ),
    # -------------------------------------------------------------- coming
    "xc.functionals": dict(
        kind="coming", summary="The exchange-correlation functionals of Libxc's LDA and GGA families, "
        "evaluated on a density grid: energy density and potential.",
        where="xc/ holds them as Viv, generated from Libxc's Maple sources by tools/vivxc and checked by "
        "xc/tests/oracle.py; they have no manifest yet."),
    "symreg.sisso": dict(
        kind="coming", summary="SISSO: a descriptor built from a feature space of formulas, chosen by sure "
        "independence screening and sparse regression.",
        where="Not in this repository yet."),
    "symreg.ai_feynman": dict(
        kind="coming", summary="AI Feynman-style symbolic regression: recover a physics formula from samples "
        "by dimensional analysis, symmetry and separability tests.",
        where="lang/viv/examples/symbolic_regression.viv searches small expression trees; it has no manifest yet."),
    "hardware.riscv": dict(
        kind="coming", summary="An RV32I instruction-set simulator: run a program and read back the "
        "registers, memory and cycle count.",
        where="The simulator is written: lib/rv32/sim.viv is an RV32IM instruction-set simulator in Viv that "
        "passes the riscv-tests rv32ui and rv32um suites, and the gallery runs a C program on it. The Python "
        "binding vp.hardware.riscv is not built yet."),
}
