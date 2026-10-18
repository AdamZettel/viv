/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2019 Rui Ueyama */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* Local variables and chained assignment (chibicc test.sh). */
int main() {
    int a;
    int b;
    a = b = 3;
    int foo123 = 3, bar = 5;
    print_int(a + b);
    print_int(foo123 + bar);
    return a * b + bar;
}
