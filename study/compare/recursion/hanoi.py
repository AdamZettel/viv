# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ move n disks from one peg to another
def hanoi(n, source, target, spare):
    if n == 0:
        return []
    moves = hanoi(n - 1, source, spare, target)
    moves.append(f"disk {n}: {source} -> {target}")
    return moves + hanoi(n - 1, spare, target, source)

# @@ three disks, every move
DISKS = 3
for m in hanoi(DISKS, "A", "C", "B"):
    print(m)

# @@ the count doubles with each disk
MAX_DISKS = 10
EVERY = 3
for n in range(1, MAX_DISKS + 1, EVERY):
    count = len(hanoi(n, "A", "C", "B"))
    print(f"{n:2d} disks: {count:4d} moves")

# @@ checks
assert len(hanoi(MAX_DISKS, "A", "C", "B")) == 2 ** MAX_DISKS - 1
