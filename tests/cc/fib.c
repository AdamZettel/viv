/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2019 Rui Ueyama */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* Recursive Fibonacci (chibicc test.sh's fib), printed for n = 0..12. */
int fib(int x) {
    if (x <= 1) return x;
    return fib(x - 1) + fib(x - 2);
}
int main() {
    int n;
    for (n = 0; n <= 12; n = n + 1)
        print_int(fib(n));
    return fib(11);
}
