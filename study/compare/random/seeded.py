# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ imports
import numpy as np

# @@ a generator from a seed
SEED = 42
rng = np.random.default_rng(SEED)

# @@ uniform draws in [0, 1)
FIRST = 4
u = rng.random(FIRST)
print("first draws", " ".join(f"{x:.6f}" for x in u))

# @@ estimate pi from random points in the unit square
N = 20000
x = rng.random(N)
y = rng.random(N)
inside = np.count_nonzero(x * x + y * y < 1.0)
print(f"pi ~ {4 * inside / N:.4f} from {N} points")

# @@ dice
ROLLS = 12
rolls = rng.integers(1, 7, size=ROLLS)
print("dice", rolls.tolist())

# @@ shuffle a deck
CARDS = 10
deck = np.arange(CARDS)
rng.shuffle(deck)
print("deck", deck.tolist())

# @@ normal draws
M = 1000
z = rng.standard_normal(M)
print(f"mean {z.mean():.4f}, std {z.std():.4f}")

# @@ checks
assert inside == np.sum(x * x + y * y < 1.0)
assert rolls.min() >= 1 and rolls.max() <= 6
