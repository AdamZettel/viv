/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2019 Rui Ueyama */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* Towers of Hanoi: the number of moves, counted by recursion. */
int moves(int n, int from, int to, int via) {
    if (n == 0) return 0;
    return moves(n - 1, from, via, to) + 1 + moves(n - 1, via, to, from);
}
int main() {
    int n;
    for (n = 1; n <= 8; n = n + 1)
        print_int(moves(n, 1, 3, 2));
    return moves(7, 1, 3, 2);
}
