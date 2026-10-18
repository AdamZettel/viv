/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
#ifdef VIV_VMATH
#include "../../lang/viv/src/vmath.c"
#define OUT "v_%s.bin"
#else
#define OUT "c_%s.bin"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
/* The platform's C libm on the sweep, or with -DVIV_VMATH Viv's own (vmath.c,
   what a compiled Viv program calls): writes each function's results */
static double *rd(const char *f, long *n) {
    FILE *fp = fopen(f, "rb"); fseek(fp, 0, SEEK_END); long sz = ftell(fp); fseek(fp, 0, SEEK_SET);
    double *d = malloc(sz); fread(d, 1, sz, fp); fclose(fp); *n = sz / 8; return d;
}
int main(void) {
    long n, m; double *x = rd("in.bin", &n); double *p = rd("pow.bin", &m);
    const char *names[] = {"sin","cos","tan","exp","log","atan","asin","acos","sinh","cosh","tanh"};
    double (*fs[])(double) = {sin,cos,tan,exp,log,atan,asin,acos,sinh,cosh,tanh};
    for (int k = 0; k < 11; k++) {
        char f[64]; snprintf(f, sizeof f, OUT, names[k]);
        double *o = malloc(n * 8);
        for (long i = 0; i < n; i++) o[i] = fs[k](x[i]);
        FILE *fp = fopen(f, "wb"); fwrite(o, 8, n, fp); fclose(fp); free(o);
    }
    double *o = malloc(m / 2 * 8);
    for (long i = 0; i < m / 2; i++) o[i] = pow(p[2*i], p[2*i+1]);
    char pf[64]; snprintf(pf, sizeof pf, OUT, "pow");
    FILE *fp = fopen(pf, "wb"); fwrite(o, 8, m / 2, fp); fclose(fp);
    return 0;
}
