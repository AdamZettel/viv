/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* viv_pow before and after the n<<20 fix: 20 million
   inputs (random bit patterns, and ranges where the result scales by 2^n with n
   negative), the results hashed. Build: cc -O2 -DVMATH='"lang/viv/src/vmath.c"' pow_bits.c -lm
   main 9bfcc15:  20000000 508024c1f54938b0  (UBSan: left shift of negative value -299)
   own1:          20000000 508024c1f54938b0  (UBSan: clean) */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include VMATH
static uint64_t st = 88172645463325252ULL;
static uint64_t xr(void) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; }
int main(void) {
    uint64_t h = 1469598103934665603ULL; long n = 0;
    for (long k = 0; k < 20000000; k++) {
        uint64_t a = xr(), b = xr();
        double x, y;
        memcpy(&x, &a, 8); memcpy(&y, &b, 8);
        if (k % 4 == 1) { x = (double)(a % 1000000) / 997.0; y = ((double)(int64_t)(b % 4000) - 2000.0) / 1.7; }
        if (k % 4 == 2) { x = 1.0 + (double)(a % 1000) * 1e-6; y = ((double)(int64_t)(b % 2000000) - 1000000.0); }
        if (k % 4 == 3) { x = (double)(a % 100) * 0.37; y = -((double)(b % 1100)); }
        double z = viv_pow(x, y);
        uint64_t u; memcpy(&u, &z, 8);
        h = (h ^ u) * 1099511628211ULL; n++;
    }
    printf("%ld %016llx\n", n, (unsigned long long)h);
    return 0;
}
