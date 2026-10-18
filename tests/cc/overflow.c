/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2019 Rui Ueyama */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* int arithmetic that does not overflow but comes close: products near
 * 2^31 and the M extension's 32-bit mul, div and rem on large values. */
int main() {
    int big = 46340;
    print_int(big * big);
    print_int(2147483647 / 7);
    print_int(2147483647 % 1000);
    print_int(-2147483647 / 3);
    print_int(-2147483647 % 3);
    print_int(65535 * 32767);
    return (big * big) % 256;
}
