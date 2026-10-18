/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2019 Rui Ueyama */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* Primes below 200 by trial division; prints each, returns the count. */
int is_prime(int n) {
    int d;
    if (n < 2) return 0;
    for (d = 2; d * d <= n; d = d + 1)
        if (n % d == 0) return 0;
    return 1;
}
int main() {
    int n;
    int count = 0;
    for (n = 0; n < 200; n = n + 1) {
        if (is_prime(n)) {
            print_int(n);
            count = count + 1;
        }
    }
    return count;
}
