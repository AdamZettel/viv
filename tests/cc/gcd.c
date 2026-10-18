/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2019 Rui Ueyama */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* Euclid's algorithm, recursive and iterative. */
int gcd(int a, int b) {
    if (b == 0) return a;
    return gcd(b, a % b);
}
int gcd_loop(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}
int main() {
    print_int(gcd(1071, 462));
    print_int(gcd_loop(1071, 462));
    print_int(gcd(17, 5));
    print_int(gcd_loop(270, 192));
    return gcd(84, 36);
}
