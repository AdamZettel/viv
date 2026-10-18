# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
A probit GLM, R's way
=====================

`vp.stats.glm` is R's `glm.fit` step for step. Here a probit model of
mtcars' engine shape (vs: straight or V) against fuel economy, the data as
R ships it. The coefficient table is R's `summary.glm`, and R 4.3.3 gives
the same numbers within 2e-15 relative. `lib/stats/glm_check.json` records
the comparison for this fit and six more of R's datasets: at R's epsilon
1e-14 the coefficients, standard errors and deviance agree within 1e-10. The fitting method and the
information behind the standard errors are named slots: IRLS with the
expected information is R's choice; Newton with the observed information
(its second derivatives taken by Viv's `deriv`) is the other.
"""

# %%
# ## The fit

import numpy as np
import vipyr as vp
from vipyr.figure import figure

# mtcars$mpg and mtcars$vs (R's datasets package)
mpg = np.array([21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4,
                22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4,
                14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3,
                19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4])
vs = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
               0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1.0])
X = np.column_stack([np.ones(32), mpg])
g = vp.stats.glm(
    vs, X, family="binomial", link="probit", names=["(Intercept)", "mpg"]
)
print(g.summary())

# %%
# ## Expected or observed information
#
# For probit the two informations differ away from the fitted means, so
# the standard errors do too; at the canonical logit link they would be
# the same matrix. Both fits run to epsilon 1e-12 here, so that the two
# methods' coefficients can be compared to more digits than R's default
# 1e-8 leaves.

g = vp.stats.glm(vs, X, family="binomial", link="probit", epsilon=1e-12)
h = vp.stats.glm(vs, X, family="binomial", link="probit",
                 fit="newton", cov="observed", epsilon=1e-12)
print("coefficients, IRLS  ", np.round(g.coef, 10))
print("coefficients, Newton", np.round(h.coef, 10))
print("SE, expected information", np.round(g.bse, 6))
print("SE, observed information", np.round(h.bse, 6))

# %%
# ## Diagnostics
#
# The fitted probability against mpg with the data, and the deviance
# residuals against the linear predictor.

grid = np.linspace(10, 34, 97)
p = vp.stats.norm.cdf(g.coef[0] + g.coef[1] * grid)
fig = figure("P(vs = 1) against mpg, probit fit", "mpg", "probability")
fig.plot(grid, p, label="fitted")
fig.scatter(mpg, vs, label="mtcars")
fig

# %%

fig = figure("Deviance residuals", "linear predictor", "deviance residual")
fig.scatter(g.eta, g.resid_deviance, label="residuals")
fig.hline("zero", 0.0)
fig
