# viv-xc

viv-xc: exchange-correlation functionals derived from Libxc 6.1.0 by
M.A.L. Marques, S. Lehtola et al.; MPL-2.0.

Each functional is one Viv module, generated from Libxc's Maple script for it
(`maple/lda_exc/*.mpl`, `maple/gga_exc/*.mpl`) by `tools/vivxc/gen.py`:

- `xc/lda/<name>.viv`, `xc/gga/<name>.viv`, named as Libxc names them
  (`gga_x_pbe` is `XC_GGA_X_PBE`, id 101). Each module has `zk_unpol(x)` and
  `zk_pol(x)`, the energy per particle as Libxc defines it, and
  `eval_unpol(rho[, sigma])` / `eval_pol(rho[, sigma])`, which return what
  `xc_lda_exc_vxc` / `xc_gga_exc_vxc` return (zk, vrho, vsigma, flattened per
  point) with Libxc's density, sigma and zeta thresholds applied the same way.
- vrho and vsigma are the reverse-mode gradient (`grad`) of the energy per
  volume, `dens * zk`: no derivative is written by hand or by the generator.
- `xc/catalog.json`: every Libxc 6.1.0 LDA and GGA functional, translated or
  skipped, with the reason.
- `xc/tests/oracle.py`: the generated modules against Libxc 6.1.0 (the copy
  PySCF 2.5 ships), on log-spaced densities, gradients and spin
  polarizations. `--smoke` is the premerge subset.

- `xc/mix/<name>.viv`: functionals Libxc builds as a weighted sum of others
  (B3LYP, PBE0 = `hyb_gga_xc_pbeh`, ...), summing the component modules.

Hybrids keep their semilocal part only; the exact-exchange fractions are
recorded in each module's doc lines (`#~ hybrid:`). Not translated (see
`catalog.json` for each one's reason): VV10, scripts needing special functions
Viv has no adjoint for (erfc, E1, LambertW, dilog), the `enforce_smooth_lr`
series procedures of the range-separated scripts, potential-only scripts, and
functionals whose parameters Libxc computes in C (`tools/vivxc/quarantine.json`).

`xc/tests/expected.json` records each module's oracle status: `pass`, `edge`
(disagrees only where one spin density is below 1e-6 of the total or a sigma
is zero, where Libxc's own result is set by rounding), or `fail`.

## License

The generated files are a Larger Work derived from Libxc's Maple sources and
are under the Mozilla Public License 2.0 (`LICENSE`, copied from Libxc's
`COPYING`). The generator (`tools/vivxc/`), `xc/core.viv` and
`xc/tests/oracle.py` are Viv's own code, MIT OR Apache-2.0.

## Citing

Cite Libxc when you use these functionals:

> Susi Lehtola, Conrad Steigemann, Micael J.T. Oliveira, and Miguel A.L.
> Marques, Recent developments in Libxc - A comprehensive library of
> functionals for density functional theory, SoftwareX 7, 1 (2018).
> doi:10.1016/j.softx.2017.11.002

and the functional's own paper, which each module lists (`#~ reference:`).
