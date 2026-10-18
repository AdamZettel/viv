---
title: Seeded random numbers
order: 1
output: identical
---
- Viv's own `srand(seed)` and `rand()` are a Park-Miller generator, not
  NumPy's. To draw the same numbers as `np.random.default_rng(42)`, the Viv
  side uses lib/nprandom, a Viv port of NumPy's PCG64 generator and its
  methods, brought in with `#! use lib/nprandom/generator`.
- The generator is an array of state that each call updates:
  `rng.random()` is `npr_pcg_double(rng)`, `rng.integers(1, 7)` is
  `1 + npr_gn_bounded(rng, 5)` (the bound is inclusive), `rng.shuffle(deck)` is
  `npr_gn_shuffle(rng, deck)` and `rng.standard_normal()` is
  `npr_gn_normal(rng)`.
- The library draws one number per call, so the vector calls
  `rng.random(N)` become loops. The draws must come in NumPy's order: all of
  `x`, then all of `y`, not the two interleaved.
- `np.count_nonzero(x * x + y * y < 1.0)` is `sum(...)` of the comparison, which
  is an array of 1.0 and 0.0.
- `npr_gn_normal` gives NaN where NumPy's value depends on the platform's
  `log1p` (about one draw in 4000 falls in that tail). The test checks that
  none of these 1000 draws did: `all(z == z)` is false if any is NaN.
- `z.std()` divides by `M`, as the Viv formula does.
