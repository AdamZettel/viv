/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2019 Rui Ueyama */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* Integer square root by bisection, and the perfect squares it finds. */
int isqrt(int n) {
    int lo = 0;
    int hi = n + 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (mid * mid <= n) lo = mid;
        else hi = mid;
    }
    return lo;
}
int main() {
    int n;
    print_int(isqrt(0));
    print_int(isqrt(99));
    print_int(isqrt(100));
    print_int(isqrt(2000000));
    for (n = 1; n < 50; n = n + 1)
        if (isqrt(n) * isqrt(n) == n) print_int(n);
    return isqrt(65535);
}
