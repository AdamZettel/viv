/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2019 Rui Ueyama */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* Nested loops: sums of squares, a multiplication table checksum and
 * triangular numbers. */
int main() {
    int i;
    int j;
    int s = 0;
    int t = 0;
    for (i = 1; i <= 100; i = i + 1) s = s + i * i;
    print_int(s);
    for (i = 1; i <= 12; i = i + 1)
        for (j = 1; j <= 12; j = j + 1)
            t = t + i * j * (i + j) % 7;
    print_int(t);
    s = 0;
    i = 0;
    while (s < 1000) {
        i = i + 1;
        s = s + i;
    }
    print_int(i);
    print_int(s);
    return t % 100;
}
