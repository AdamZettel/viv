/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2019 Rui Ueyama */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* if, else, for and while (chibicc test.sh), with the sums printed. */
int main() {
    int i = 0;
    int j = 0;
    for (i = 0; i <= 10; i = i + 1)
        j = i + j;
    print_int(j);
    i = 0;
    while (i < 10)
        i = i + 1;
    print_int(i);
    if (i == 10) print_int(1); else print_int(2);
    if (i != 10) print_int(3); else if (j == 55) print_int(4); else print_int(5);
    for (;;) {
        if (i > 15) return i;
        i = i + 2;
    }
    return 0;
}
