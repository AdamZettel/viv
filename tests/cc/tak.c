/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2019 Rui Ueyama */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* Takeuchi's function, a classic benchmark of calls and comparisons. */
int tak(int x, int y, int z) {
    if (y < x)
        return tak(tak(x - 1, y, z), tak(y - 1, z, x), tak(z - 1, x, y));
    return z;
}
int main() {
    print_int(tak(6, 4, 2));
    print_int(tak(9, 6, 3));
    return tak(8, 4, 0);
}
