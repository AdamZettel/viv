# Option pricing in Viv, called from Python

Six Viv modules that price options, each imported from Python like a Python
module, each checked against closed forms, published values and independent
NumPy/SciPy code.

| module | what it computes |
|---|---|
| `bs.viv` | Black-Scholes prices and implied volatilities for arrays of quotes |
| `crr.viv` | Cox-Ross-Rubinstein binomial trees, European and American |
| `heston.viv` | European options under Heston by the COS method (complex arithmetic) |
| `cnpde.viv` | Crank-Nicolson: American put, down-and-out call |
| `mc.viv` | Monte Carlo on Philox4x32-10: arithmetic Asian with a control variate, Longstaff-Schwartz American put |
| `greeks.viv` | delta, gamma and vega by Viv's autodiff, for Black-Scholes and Heston |

## The workflow

Write the slow function in a `.viv` file beside your Python, with typed
parameters and test blocks:

```
def bs_price(S: Num, K: Num, T: Num, r: Num, q: Num, sigma: Num, is_call: Bool) -> Num {
    ...
}

test "Hull's example: S=42, K=40, r=10%, sigma=20%, T=0.5 (call 4.76, put 0.81)" {
    assert_near(bs_price(42.0, 40.0, 0.5, 0.1, 0.0, 0.2, true), 4.7594, 5.0e-5)
}
```

Run its tests with `vipyr viv run bs.viv`. Then import it:

```python
import vipyr.auto      # once, before the first .viv import
import bs                # compiles bs.viv to a shared library, cached

iv = bs.implied_vols(prices, S, K, T, 0.02, 0.0, is_call)   # NumPy arrays in and out
```

The first import compiles the file (a second or two); later imports load
the cached library. Arrays cross without a copy when the Viv function only
reads them: the library reads the NumPy buffer itself. A function that
writes into an argument gets its own copy, so the caller's array never
changes.

Run the tests against the references with
`python -m pytest examples/options/test_options.py` (24 tests, about 5 s).

## What was checked

| kernel | reference | worst error | tolerance |
|---|---|---:|---:|
| normal CDF (Cody) | mpmath, 40 digits, x in [-37.5, 8] | 7.7e-16 rel | 1e-15 |
| Black-Scholes price | SciPy closed form, 2000 random quotes | 1.0e-13 rel | 1e-12 |
| implied vol round trip | 1080-quote grid; 898 with time value > 1e-5 of the price | 8.8e-13 rel | 1e-12 |
| CRR American put | Hull's printed values 4.49, 4.263, 4.272, 4.278, 4.283 (N = 5 ... 500) | within the printed digits | half a unit of the last digit |
| CRR European, N = 4000 | Black-Scholes | 5.4e-5 | 2e-3 |
| Heston COS | Fang-Oosterlee (2008) 5.785155450 (T = 1), 22.318945791 (T = 10) | 1.6e-8, 1.5e-10 | 2e-8, 1e-9 |
| Heston COS | SciPy `quad` of the Gil-Pelaez integrals, 21 options | 2.0e-13 | 1e-9 |
| Crank-Nicolson American put | CRR, mean of N = 10000 and 10001 (4.48669) | 8.8e-5 | 2e-4 |
| Crank-Nicolson down-and-out call | Reiner-Rubinstein closed form, 3 cases | 5.6e-5 | 2e-3 |
| Philox4x32-10 | Random123 known-answer vectors; a Python implementation, 200 random counters | exact | exact |
| geometric Asian (MC) | closed form | 1.5 s.e. | 4 s.e. |
| arithmetic Asian, control variate | NumPy twin on the same Philox stream | 1.6e-13 | 1e-10 |
| Longstaff-Schwartz put | binomial 4.48669 (LS 2001 report 4.472, s.e. 0.010) | 4.4654, s.e. 0.0092 | 3 s.e. + 0.01 |
| Black-Scholes delta, gamma, vega | closed forms, 300 options | 2.2e-15 abs, 2.7e-13 rel, 2.7e-13 rel | 1e-13, 1e-12, 1e-12 |
| Heston delta, gamma, dV/dv0 | central differences of the SciPy price | 1.6e-7, 1.8e-7, 8.9e-6 | 1e-6, 1e-5, 1e-5 |

Notes on the edges:

- **Implied volatility** is a Newton iteration on log(price) in
  sigma sqrt(T), from the out-of-the-money side of put-call parity, kept in
  a bracket that falls back to bisection. Not Jäckel's "Let's Be Rational";
  it converges in a few steps on the grid and returns NaN for a price
  outside the no-arbitrage bounds. Deep out-of-the-money quotes (prices
  near 1e-40) invert to 1e-12. Deep in-the-money quotes lose accuracy in
  proportion to price / time value: a double holding the price cannot pin
  a time value of 1e-10 of it, and no solver can recover what the rounding
  removed.
- **The published Heston value** 5.785155450 is 1.6e-8 from what COS and
  `quad` agree on to 1e-12 (5.78515543438). The T = 10 value agrees to
  1.5e-10.
- **Longstaff and Schwartz's own finite-difference value** for their first
  case, 4.478, is 0.009 below the converged binomial value; the PDE here is
  compared with the binomial value.
- **Gamma** is not a second derivative by autodiff: Viv's `grad` does not
  yet take the gradient of a gradient. Black-Scholes gamma uses the model's
  identity gamma = vega / (S^2 sigma T), with vega by autodiff. Heston gamma
  is a central difference of the autodiff delta.
- **Heston Greeks** run on a real-arithmetic copy of the COS sum, since
  `grad` does not differentiate complex numbers.

## Speed

From [RESULTS.md](RESULTS.md) (Apple M2 Pro, median of 5 warm runs, load
below 4):

| benchmark | pure Python | NumPy | Numba | Viv exact | Viv fast |
|---|---:|---:|---:|---:|---:|
| implied vol, 1e6 quotes | 7.79 s | 4.52 s | 0.585 s | 0.863 s | 0.568 s |
| CRR American, 1e3 options, N = 1000 | 118 s | 10.8 s | 0.571 s | 0.897 s | 0.688 s |
| Crank-Nicolson, 2000 x 1000 grid | 1.38 s | 1.04 s | 0.029 s | 0.042 s | 0.036 s |
| MC Asian, 1e6 paths x 12 dates | 22.0 s | 0.660 s | 0.258 s | 1.42 s | 1.22 s |

Numba is faster than the exact Viv build on all four. Viv checks every
array index and every integer operation, and it has no integer xor or shift
yet, so the Monte Carlo benchmark, mostly Philox, runs 5.5x slower than
Numba. RESULTS.md has the
details.

## Same bits on every machine

The exact build (the default) compiles with fused multiply-add off and calls
Viv's own `exp`, `log`, `sin` and `cos` instead of the platform's, so a
result does not depend on the CPU, the operating system or the C library: a
price computed on a laptop matches the one computed on a server to the last
bit, and matches `vipyr viv run` in Viv's interpreter. The fast build
gives up that guarantee for speed.

The Monte Carlo goes one step further. Philox is counter-based: the normals
of path p come from the counter (p, block) and the seed, not from a
generator's running state. The paths are drawn in a `parallel for`, each
writes only its own slot, and every sum is taken afterward in path order.
`test_bit_identical_across_thread_counts` builds `mc.viv` with OpenMP and
gets the same SHA-256 of the results on 1, 2, 4 and 8 threads.
