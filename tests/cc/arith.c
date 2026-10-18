/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2019 Rui Ueyama */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* Arithmetic, precedence and comparisons: expressions from chibicc's
 * test.sh and tests/arith.c (github.com/rui314/chibicc). */
int main() {
    print_int(5 + 20 - 4);
    print_int(5 + 6 * 7);
    print_int(5 * (9 - 6));
    print_int((3 + 5) / 2);
    print_int(-10 + 20);
    print_int(- -10);
    print_int(17 % 6);
    print_int(-17 % 6);
    print_int(-17 / 6);
    print_int(0 == 1);
    print_int(42 != 42);
    print_int(1 >= 1);
    print_int(!0 + !5);
    print_int(0 || (2 - 2) || 5);
    print_int((2 - 2) && 5);
    return 21;
}
