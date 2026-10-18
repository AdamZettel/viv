/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* Every float32 input of one function.
   Viv's Float32 exp, log, sin, cos and tanh are Viv's double function
   (vmath.c, the same bits in every engine) rounded once to float32. That
   rounding is the correctly rounded float32 unless the double result lies
   within the double function's error of a float32 rounding boundary (a
   midpoint between two adjacent float32s). This program visits all 2^32
   inputs and writes each input whose double result lies within K double
   ulps of a midpoint; check.py then decides those with mpmath. K is 1 for
   exp (viv_exp_cr is correctly rounded: error at most half an ulp) and 4
   for the others (fdlibm's documented bound is below 1 ulp).

       cc -O2 -ffp-contract=off exhaustive.c -o exhaustive -lm
       ./exhaustive exp > exp.hard      (one function, one thread)

   Output lines: the input's bits in hex, the double result's bits in hex.

       ./exhaustive exp sample 20000    the same lines for 20000 random
                                        inputs, for check.py --bound */
#include "../../lang/viv/src/vmath.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double ulp(double y) {
    double a = fabs(y);
    return nextafter(a, INFINITY) - a;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: exhaustive exp|log|sin|cos|tanh\n"); return 2; }
    double (*f)(double) = 0;
    double k = 4.0;
    if (!strcmp(argv[1], "exp")) { f = viv_exp_cr; k = 1.0; }
    else if (!strcmp(argv[1], "log")) f = viv_log;
    else if (!strcmp(argv[1], "sin")) f = viv_sin;
    else if (!strcmp(argv[1], "cos")) f = viv_cos;
    else if (!strcmp(argv[1], "tanh")) f = viv_tanh;
    else { fprintf(stderr, "unknown function %s\n", argv[1]); return 2; }
    if (argc > 3 && !strcmp(argv[2], "sample")) {
        /* a 64-bit LCG (Knuth's MMIX constants): the same inputs every run */
        unsigned long long st = 1, n = strtoull(argv[3], 0, 10);
        for (unsigned long long i = 0; i < n; i++) {
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            uint32_t b = (uint32_t)(st >> 32);
            float x; memcpy(&x, &b, 4);
            if (x != x) continue;
            double y = f((double)x);
            uint64_t yb; memcpy(&yb, &y, 8);
            printf("%08x %016llx\n", b, (unsigned long long)yb);
        }
        return 0;
    }
    unsigned long long hard = 0, visited = 0;
    for (unsigned long long u = 0; u <= 0xffffffffULL; u++) {
        uint32_t b = (uint32_t)u;
        float x; memcpy(&x, &b, 4);
        if (x != x) continue;
        visited++;
        double y = f((double)x);
        if (!(fabs(y) <= FLT_MAX * 2.0) || y == 0.0) continue;
        float r = (float)y;
        if (isinf(r)) {
            /* past the overflow threshold, 2^128 - 2^103: near it is hard */
            double t = 340282356779733661637539395458142568448.0;
            if (fabs(fabs(y) - t) <= k * ulp(y)) {
                uint64_t yb; memcpy(&yb, &y, 8);
                printf("%08x %016llx\n", b, (unsigned long long)yb);
                hard++;
            }
            continue;
        }
        /* the neighbour of r on y's side, and the midpoint between them
           (exact in a double: 25 significant bits) */
        if ((double)r == y) continue;
        float s = (double)r < y ? nextafterf(r, INFINITY) : nextafterf(r, -INFINITY);
        double m = ((double)r + (double)s) / 2.0;
        if (isinf(s)) m = 340282356779733661637539395458142568448.0 * (r < 0 ? -1.0 : 1.0);
        if (fabs(y - m) <= k * ulp(y)) {
            uint64_t yb; memcpy(&yb, &y, 8);
            printf("%08x %016llx\n", b, (unsigned long long)yb);
            hard++;
        }
    }
    fprintf(stderr, "%s: %llu inputs, %llu within %g ulp of a float32 midpoint\n", argv[1], visited, hard, k);
    return 0;
}
