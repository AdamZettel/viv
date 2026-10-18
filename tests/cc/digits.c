/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2019 Rui Ueyama */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* Digit sums, digit reversal and palindromes: / and % on negatives too. */
int digit_sum(int n) {
    int s = 0;
    if (n < 0) n = -n;
    while (n > 0) {
        s = s + n % 10;
        n = n / 10;
    }
    return s;
}
int reverse(int n) {
    int r = 0;
    while (n != 0) {
        r = r * 10 + n % 10;
        n = n / 10;
    }
    return r;
}
int main() {
    int n;
    int pal = 0;
    print_int(digit_sum(987654321));
    print_int(digit_sum(-4096));
    print_int(reverse(12345));
    print_int(reverse(-120));
    for (n = 100; n < 200; n = n + 1)
        if (reverse(n) == n) pal = pal + 1;
    print_int(pal);
    return digit_sum(99999);
}
