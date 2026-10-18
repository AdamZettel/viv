/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* Links Viv's libm into the comparison builds of upstream, so that the
   Fortran and the port call the same exp, log, pow, sin, cos and atan
   (sqrt is correctly rounded everywhere). As ports/dftatom/oracle does.
   gfortran turns the SIN and COS of one argument (random_normal's
   Box-Muller) into one call of cexp on an imaginary number, so cexp is
   here too, as e^x (cos y + i sin y) with Viv's functions.

   cc -O2 -ffp-contract=off -c -I<viv>/lang/viv/src vmath_shim.c   */
#include "vmath.c"
#undef exp
#undef log
#undef pow
#undef sin
#undef cos
#undef atan
#undef cexp
double exp(double x) { return viv_exp_cr(x); }
double log(double x) { return viv_log(x); }
double pow(double x, double y) { return viv_pow_sq(x, y); }
double sin(double x) { return viv_sin(x); }
double cos(double x) { return viv_cos(x); }
double atan(double x) { return viv_atan(x); }
double _Complex cexp(double _Complex z)
{
    double e = viv_exp_cr(__real__ z);
    double _Complex w;
    __real__ w = e * viv_cos(__imag__ z);
    __imag__ w = e * viv_sin(__imag__ z);
    return w;
}
