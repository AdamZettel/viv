# vipyr-xc

A distribution of [vipyr](../../python/vipyr/README.md), Viv as a Python
package. Libxc's LDA and GGA functionals as Viv (the repository's xc/ directory). Listed, not wrapped yet.

    pip install vipyr[xc]

```python
import vipyr as vp
vp.functions()          # every function vipyr can see, this distribution's included
```

Installing it registers `vp.xc` in the entry-point group `vipyr.packages`;
`import vipyr as vp` then finds it on first use. Without it,
`vp.xc` raises an ImportError that names this distribution.

The functions, their parameters and results are generated with the rest:
`python -m vipyr.docs` writes `vipyr_xc/spec.json` from the programs'
manifests. Reference: https://viv-lang.org/vipyr.

License: MPL-2.0 (Libxc's license; see LICENSE).
