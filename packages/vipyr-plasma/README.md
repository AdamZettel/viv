# vipyr-plasma

A distribution of [vipyr](../../python/vipyr/README.md), Viv as a Python
package. Kinetic simulations of a collisionless plasma (`vp.plasma.landau`).

    pip install vipyr[plasma]

```python
import vipyr as vp
vp.functions()          # every function vipyr can see, this distribution's included
```

Installing it registers `vp.plasma` in the entry-point group `vipyr.packages`;
`import vipyr as vp` then finds it on first use. Without it,
`vp.plasma` raises an ImportError that names this distribution.

The functions, their parameters and results are generated with the rest:
`python -m vipyr.docs` writes `vipyr_plasma/spec.json` from the programs'
manifests. Reference: https://viv-lang.org/vipyr.

License: MIT OR Apache-2.0 (LICENSE).
