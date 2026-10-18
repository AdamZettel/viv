# vipyr-liquids

The Lennard-Jones programs of **Allen and Tildesley's examples**
(https://github.com/Allen-Tildesley/examples, by Michael P. Allen and
Dominic J. Tildesley, CC0), which accompany *Computer Simulation of
Liquids*, 2nd ed. (Oxford University Press, 2017), ported to Viv and
callable from Python. If you use them, cite the book; for the equation of
state, cite Thol et al. as well (CITATION.cff).

```python
import vipyr as vp

eq = vp.liquids.md_nvt(n=256, density=0.75, temperature=1.0, nblock=4, nstep=1000)
r = vp.liquids.md_nve(r_start=eq.r_final, v_start=eq.v_final, box_start=eq.box,
                      nblock=10, nstep=2000, sample=5, nt=200)
print(r.e_cs, r.p_cs, r.p_cs_err, r.t_kin, r.d_vacf, r.d_msd)
print(vp.liquids.eos(temperature=r.t_kin, density=0.75).p_cs)
```

| Call | Upstream | What |
|---|---|---|
| `vp.liquids.md_nve` | md_nve_lj (+ _ll, _vl) | molecular dynamics, constant energy |
| `vp.liquids.md_nvt` | md_nvt_lj | molecular dynamics, Nose-Hoover chain thermostat |
| `vp.liquids.mc_nvt` | mc_nvt_lj | Metropolis Monte Carlo, constant volume |
| `vp.liquids.mc_npt` | mc_npt_lj | Metropolis Monte Carlo, constant pressure |
| `vp.liquids.eos` | eos_lj | Thol et al.'s equation of state |
| `vp.liquids.pair_distribution` | pair_distribution | g(r) of configurations |
| `vp.liquids.diffusion` | diffusion | VACF, MSD and D of a trajectory |
| `vp.liquids.error_calc` | error_calc | errors of correlated averages |

Every program gives the same doubles as upstream's Fortran from the same
start and random numbers (COMPARISON.md); UPSTREAM.md lists what was ported
and how the port differs.

    python3 ports/liquids/test_liquids.py
