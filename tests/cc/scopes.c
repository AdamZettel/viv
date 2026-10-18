/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2019 Rui Ueyama */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* Blocks, null statements, early returns, and a deep expression stack. */
int deep(int a) {
    return a + (a + (a + (a + (a + (a + (a + (a + (a + (a + 1)))))))));
}
int main() {
    int a = 3;
    ;;
    {
        int b = 4;
        {
            int c = a * b;
            print_int(c);
        }
        ;
    }
    print_int(deep(2));
    print_int(((((1 + 2) * (3 + 4)) - ((5 - 6) * (7 - 8))) / 2) % 7);
    return deep(a);
}
