# vipyr-nuclear

A distribution of [vipyr](../../python/vipyr/README.md), Viv as a Python
package. Skyrme Hartree-Fock (`vp.nuclear.skyrme_hf`) and the nuclear shell model (`vp.nuclear.shell_model`).

    pip install vipyr[nuclear]

```python
import vipyr as vp
vp.functions()          # every function vipyr can see, this distribution's included
```

Installing it registers `vp.nuclear` in the entry-point group `vipyr.packages`;
`import vipyr as vp` then finds it on first use. Without it,
`vp.nuclear` raises an ImportError that names this distribution.

The functions, their parameters and results are generated with the rest:
`python -m vipyr.docs` writes `vipyr_nuclear/spec.json` from the programs'
manifests. Reference: https://viv-lang.org/vipyr.

License: MIT OR Apache-2.0 (LICENSE). The shell model follows BIGSTICK (MIT); its notice is in NOTICE.
