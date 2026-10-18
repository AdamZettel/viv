/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2019 Rui Ueyama */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* The Ackermann function: deep recursion through two parameters. */
int ack(int m, int n) {
    if (m == 0) return n + 1;
    if (n == 0) return ack(m - 1, 1);
    return ack(m - 1, ack(m, n - 1));
}
int main() {
    int m;
    int n;
    for (m = 0; m <= 2; m = m + 1)
        for (n = 0; n <= 3; n = n + 1)
            print_int(ack(m, n));
    return ack(2, 3);
}
