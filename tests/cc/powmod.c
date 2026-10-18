/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2019 Rui Ueyama */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* Modular exponentiation by squaring; Fermat's little theorem checked. */
int powmod(int b, int e, int m) {
    int r = 1;
    b = b % m;
    while (e > 0) {
        if (e % 2 == 1) r = r * b % m;
        b = b * b % m;
        e = e / 2;
    }
    return r;
}
int main() {
    print_int(powmod(2, 10, 1000));
    print_int(powmod(3, 200, 1009));
    print_int(powmod(7, 1008, 1009));
    print_int(powmod(12345, 6789, 9973));
    return powmod(5, 3, 13);
}
