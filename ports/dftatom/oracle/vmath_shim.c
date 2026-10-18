/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2006-2012 Ondřej Čertík, Jiří Vackář, John Pask */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* Links Viv's libm into the upstream oracle, so that the Fortran and the
   port call the same exp, log, pow and atan (and sqrt, which IEEE rounds
   correctly everywhere). With the platform's libm the two differ by an ulp
   in a few exp calls (Viv's exp is correctly rounded, the platform's is
   not), and the SCF loop carries that into the tenth digit of a heavy
   atom's energy; with this shim, any difference left is the port's.

   cc -O2 -ffp-contract=off -c -I<viv>/lang/viv/src vmath_shim.c   */
#include "vmath.c"
#undef exp
#undef log
#undef pow
#undef atan
double exp(double x) { return viv_exp_cr(x); }
double log(double x) { return viv_log(x); }
double pow(double x, double y) { return viv_pow_sq(x, y); }
double atan(double x) { return viv_atan(x); }
