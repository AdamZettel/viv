# vipyr-chem

A distribution of [vipyr](../../python/vipyr/README.md), Viv as a Python
package. Hartree-Fock for molecules in Gaussian basis sets (`vp.scf.run`) and plane-wave density-functional theory of crystals (`vp.dft.pw`).

    pip install vipyr[chem]

```python
import vipyr as vp
vp.functions()          # every function vipyr can see, this distribution's included
```

Installing it registers `vp.scf`, `vp.dft` in the entry-point group `vipyr.packages`;
`import vipyr as vp` then finds them on first use. Without it,
`vp.scf` raises an ImportError that names this distribution.

The functions, their parameters and results are generated with the rest:
`python -m vipyr.docs` writes `vipyr_chem/spec.json` from the programs'
manifests. Reference: https://viv-lang.org/vipyr.

License: MIT OR Apache-2.0 (LICENSE).
