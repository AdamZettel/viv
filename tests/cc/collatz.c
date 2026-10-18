/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2019 Rui Ueyama */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* Collatz steps: the longest chain below 200 and where it starts. */
int steps(int n) {
    int k = 0;
    while (n != 1) {
        if (n % 2 == 0) n = n / 2;
        else n = 3 * n + 1;
        k = k + 1;
    }
    return k;
}
int main() {
    int best = 0;
    int arg = 0;
    int n;
    for (n = 1; n < 200; n = n + 1) {
        int s = steps(n);
        if (s > best) {
            best = s;
            arg = n;
        }
    }
    print_int(steps(27));
    print_int(arg);
    print_int(best);
    return best % 256;
}
