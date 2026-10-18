/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2019 Rui Ueyama */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* && and || evaluate their right operand only when it decides the result;
 * the assignments inside them show which ones ran. */
int main() {
    int x = 0;
    int y = 0;
    int r;
    r = 0 && (x = 1);
    print_int(r);
    print_int(x);
    r = 1 || (y = 1);
    print_int(r);
    print_int(y);
    r = 1 && (x = 5);
    print_int(r);
    print_int(x);
    r = 0 || (y = 0);
    print_int(r);
    print_int(y);
    return !(x == 5) + 2 * (y == 0);
}
