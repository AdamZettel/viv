/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2019 Rui Ueyama */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* Calls with up to eight arguments, nested in each other's arguments
 * (chibicc test.sh, with six-argument add6 widened to eight). */
int add2(int x, int y) { return x + y; }
int sub2(int x, int y) { return x - y; }
int add6(int a, int b, int c, int d, int e, int f) {
    return a + b + c + d + e + f;
}
int add8(int a, int b, int c, int d, int e, int f, int g, int h) {
    return a + b + c + d + e + f + g + h;
}
int weigh8(int a, int b, int c, int d, int e, int f, int g, int h) {
    return a - 2 * b + 3 * c - 4 * d + 5 * e - 6 * f + 7 * g - 8 * h;
}
int main() {
    print_int(add2(3, 4));
    print_int(sub2(4, 3));
    print_int(add6(1, 2, add6(3, 4, 5, 6, 7, 8), 9, 10, 11));
    print_int(add6(1, 2, add6(3, add6(4, 5, 6, 7, 8, 9), 10, 11, 12, 13), 14, 15, 16));
    print_int(add8(1, 2, 3, 4, 5, 6, 7, 8));
    print_int(weigh8(8, 7, 6, 5, 4, 3, 2, 1));
    print_int(weigh8(add2(1, 1), 0, 0, 0, 0, 0, 0, sub2(0, 1)));
    return add2(40, 2);
}
