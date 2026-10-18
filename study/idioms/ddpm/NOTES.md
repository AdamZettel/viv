# DDPM in Viv

Source: https://github.com/labmlai/annotated_deep_learning_paper_implementations @
`33ab02281c2b928e6b32792909cc79cbdcfe1d6a` (shallow clone at `viv-wt/src/ddpm`).
Ported: `labml_nn/diffusion/ddpm/__init__.py` (`DenoiseDiffusion`: the schedule, `q_xt_x0`,
`q_sample`, `p_sample`, `loss`), `utils.py` (`gather`), `unet.py` (`TimeEmbedding`, `Swish`), and
the `sample`/`train` loops of `experiment.py`. The UNet is replaced by a small MLP noise predictor,
and the data is a 2D swiss roll instead of images.

## Files

| file | what |
|---|---|
| `ddpm.viv` | the port: schedule, MLP with labml's time embedding, hand-written backward, Adam via `lib/optim`, ancestral sampling, statistics, figure, 7 tests |
| `ref.py` | imports the **unmodified** labml `DenoiseDiffusion`, `TimeEmbedding`, `Swish` from the clone; same MLP in torch float64; replays Viv's `rand()` stream; writes `ddpm_ref.viv` |
| `ddpm_ref.viv` | GENERATED (298 KB): torch initial weights, torch schedule, step-0 forward/loss/gradient, weights after one Adam step, loss averages, final weights, the 512 samples |
| `interp.out`, `native.out` | the two engines' output (byte-identical) |
| `reference_output.txt` | ref.py's printout |
| `probe/` | the small programs behind the friction entries |

Run: `python3 ref.py` (5 s), then `viv run ./ddpm.viv` (the `./` matters, F13).

## Sizes

| | value | labml |
|---|---|---|
| diffusion steps `T`, schedule | 1000, `beta = linspace(1e-4, 0.02)` | same |
| time embedding | `TimeEmbedding(32)`: 4 frequencies, sin+cos (8), Linear 8->32, Swish, Linear 32->32 | same class, n_channels 64 x 4 in the UNet |
| eps model | `swish(Lin(x) + temb)`, `swish(Lin 32->32)`, `Lin 32->2`; 2,562 parameters | UNet |
| training | 3000 Adam steps, batch 128, lr 1e-3, fresh roll points each step | Adam lr 2e-5 on images |
| sampling | 512 points, 1000 `p_sample` steps (noise added at every step, t = 0 included, as labml does) | same loop |

To reuse labml's code unmodified, the 2D points travel through `DenoiseDiffusion` shaped
`(B, 2, 1, 1)`: `gather` reshapes the schedule to `(-1, 1, 1, 1)`, which only broadcasts against
image-shaped tensors.

| engine | time |
|---|---|
| PyTorch reference (float64, CPU) | 4.7 s |
| Viv interpreter | 34 s |
| Viv -> C, `cc -O2` | 2.3 s |

## Agreement

`ref.py` reimplements Viv's `srand`/`rand` (Park-Miller MINSTD, exact in integers) and Box-Muller,
so both programs draw the same batches, steps `t`, training noise and sampling noise. The torch
side feeds them to labml through `torch.randint` (in `loss`) and `torch.randn` (in `p_sample`),
patched for the duration of each call. Everything is float64. Measured misses, printed by
`ddpm.viv` and asserted there:

| quantity | measured max abs miss | test tolerance |
|---|---|---|
| `beta` (1000) vs `torch.linspace` | 3.5e-18 (1 ulp at index 786; torch uses the two-sided linspace formula) | 1e-17 |
| `alpha_bar` (1000) vs `torch.cumprod` | 5.6e-17 | 1e-15 |
| step-0 `eps_theta` (128 x 2) | 1.1e-16 | 1e-13 |
| step-0 loss (1.07874) | 4.4e-16 | 1e-14 |
| step-0 gradient, all 2,562 parameters (hand backward vs autograd) | 8.3e-17 | 1e-14 |
| weights after one Adam step (`lib/optim` vs `torch.optim.Adam`) | 2.8e-17 | 1e-15 |
| 6 loss averages over 3000 steps | 5.6e-17 | 1e-13 |
| final weights after 3000 steps | 1.2e-15 | 1e-13 |
| 512 samples after 1000 sampling steps | 3.1e-15 | 1e-12 |

**Both engines:** interpreter and native C print byte-identical output (`cmp interp.out
native.out`); all 7 tests pass in each. The figure (`figure` + two `scatter`) prints as records
in both.

### What the model learned (same numbers in Viv and PyTorch)

Loss averages per 500 steps: 0.339, 0.273, 0.272, 0.266, 0.263, 0.265.

| cloud | mean (x, y) | spread (x, y) | mean distance to the roll | fraction within 0.1 |
|---|---|---|---|---|
| 512 samples | (0.352, 0.049) | (0.932, 0.950) | 0.283 | 0.199 |
| 512 target points | (0.318, -0.040) | (0.949, 0.976) | 0.0016 | 1.000 |
| 512 standard normal points | (-0.030, 0.014) | (1.012, 1.040) | 0.390 | 0.164 |

At this size the samples get the first two moments right and move toward the roll (0.28 vs 0.39),
but most points are not on it: the model is underfit. The loss has flattened at 0.26, and a
32-wide MLP with 4 time frequencies after 3000 steps is small for this target. The planned next
step was a wider model (64) and more steps (6000 to 12000), tried first in the reference, but those
runs were refused by the session's permission classifier (machine-load rule), so this document
reports the 3000-step, 32-wide run only. The port and the reference agree at any size; only the
sample quality would change.

## Line counts (code lines: no blanks, comments, docstrings)

| part | labml / ref.py | Viv |
|---|---|---|
| schedule + `DenoiseDiffusion.__init__` + sizes | ~12 | 27 (includes the 10-line parameter layout) |
| `TimeEmbedding` + `Swish` + MLP `forward` | 19 + ~12 | 48 (includes 5 layer helpers and the backward of a layer) |
| `q_sample`, `loss`, `gather` | ~15 | 43 (includes the 20-line forward+backward `loss_and_grad`) |
| `p_sample` + sampling loop | ~14 | 16 |
| training loop | ~8 | 16 |
| **ported part total** | **~80** | **150** |
| gaussian and swiss-roll generators | (torch.randn, sklearn) | 26 |
| statistics (mean, spread, distance to roll) | 8 in ref.py | 37 |

Most of the difference is the manual backward pass (F1), the bias helpers (F2), and the random
number helpers (F3).

## Friction ledger

**F1. No reverse-mode autodiff over arrays: the backward pass is written by hand, and the forward
is written twice.** sev 2, missing builtin, 1 model (10 adjoint lines + 3 helpers + a duplicated forward)
(a) had to write
```
def loss_and_grad(theta: Arr, grad: Arr, x0: Arr, t: Arr, eps: Arr) {
    fill(grad, 0.0)
    ...                                  # the forward again, keeping u, a, z1, h1, z2, h2
    dout = 2.0 * (out - eps) / (len(x0) * 2)
    dh2 = layer_back(theta, grad, h2, dout, OUT_W, OUT_B)
    dz2 = dh2 * swish_slope(z2)
    dh1 = layer_back(theta, grad, h1, dz2, HID_W, HID_B)
    dz1 = dh1 * swish_slope(z1)
    layer_back(theta, grad, xt, dz1, IN_W, IN_B)
    da = layer_back(theta, grad, a, dz1, TE2_W, TE2_B)
    layer_back(theta, grad, feats, da * swish_slope(u), TE1_W, TE1_B)
    return mse(out, eps)
}
```
plus `swish_slope`, `column_sums`, `layer_back`, and `eps_model` for inference, which repeats the
forward. Changing the architecture means changing it in two places and re-deriving adjoints.
(b) wished: `loss, grad = value_and_grad(fn(th) -> mse(eps_model(th, q_sample(x0, t, eps), t), eps))(theta)`
(c) Python: `loss = diffusion.loss(x0); loss.backward()`. Julia: `Zygote.withgradient(m -> loss(m, x0), model)`.
`deriv` is scalar forward mode only (v4 "automatic reverse transform" is future).
The manual backward did match autograd to 8e-17, and a test checks it.

**F2. No broadcasting of a row against a matrix, and no reduction along an axis.** sev 2, missing builtin, 2 helpers used at 10 sites
(a)
```
def plus_bias(X: Arr, b: Arr) {
    return X + ones_column(len(X)) @ b        # an outer product to repeat a row
}
def column_sums(G: Arr) {
    return transpose(ones_column(len(G))) @ G
}
```
`zeros(3, 2) + [1.0, 2.0]` stops with `array shape mismatch: [3, 2] vs [2]` (probe/p2.viv).
(b) `X @ W + b` and `sum(G, axis: 0)`.
(c) `x @ W.T + b`, `G.sum(0)`; Julia `X * W .+ b`, `sum(G, dims=1)`.

**F3. No normal random numbers.** sev 2, missing builtin, 3 call sites
(a) Box-Muller by hand, then a loop to fill a matrix:
```
def randn() {
    u1 = rand()
    u2 = rand()
    return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2)
}
def randn_matrix(rows, cols) { ...double loop calling randn()... }
```
(b) `Z = randn(rows, cols)`. (c) `torch.randn(rows, cols)`, `np.random.standard_normal`, Julia `randn(rows, cols)`.
Also no `rand(n)` for a vector of uniforms and no `randint`: `t[i] = floor(rand() * T)` in a loop.
A DDPM draws normal noise in every training step and every sampling step; nothing in `lib/` provides it.

**F4. Loop variables leak into global scope, and the C backend then refuses a later array of the
same name that the interpreter accepts.** sev 2, engine gap, 1
(a) The schedule loop was `for t to T { alpha_bar[t] = ... }`, and the training code later sets
`t = draw_steps(BATCH)`. The interpreter ran the program; the C backend stopped with
`` `t` is a number but is assigned a array ``. I renamed the loop variable to `s`.
(b) a loop variable scoped to its loop, or at least the same verdict in both engines.
(c) Julia scopes a `for` variable to the loop; Python leaks it but lets it be rebound.
probe/p8.viv: the interpreter prints `s: [0, 0]`, C refuses.

**F5. Error line numbers are shifted by the `#! use` modules.** sev 2, tooling/error message, every compile error
(a) The C-backend error above was reported at `./ddpm.viv:2591`; ddpm.viv has 426 lines. The line
number counts the concatenated prelude (lib/params, lib/optim, lib/polar, lib/linalg and the
298 KB ddpm_ref.viv). probe/p11.viv: a 2-line module moves the reported line from 5 to 8. The
quoted source line and caret are correct, so the error can be found by searching for the text.
(b) `./ddpm.viv:61`. (c) Python and Julia report the file and line of the module.

**F6. No slices or views: every weight read copies it out of the flat parameter vector.** sev 2, performance + missing data structure, 15 copies per training step, 5 per sampling step
(a) `param_get(theta, model, W)` (a scalar double loop in lib/params) on every forward and
backward use, and `param_add` for every gradient. probe/p10.viv: 3000 steps of three
32x32 `param_get` + `param_add` and one Adam step over 1,056 numbers take 13 s in the interpreter.
In ddpm.viv the bookkeeping (copies, `param_add`, and `adam_step`'s scalar loop over 2,562 numbers)
is most of the 34 s; the matrix products are a small share (probe/p5.viv: 1000 products of 512x32
by 32x32 plus a swish in 0.26 s).
(b) `W = theta[layout[k]]` as a view, and Adam as whole-vector expressions.
(c) `model.parameters()` are views into storage; `torch.optim.Adam` is vectorized.
Native C runs the same program in 2.3 s, so this is an interpreter cost.

**F7. No column slice.** sev 1, syntax, 6 sites
(a) `def column(P: Arr, j)` (a copy loop), then `column(samples, 0)` for every `scatter` and statistic.
`A[0:2]` is a parse error (`expected ]`, probe/p3.viv).
(b) `samples[:, 0]`. (c) `samples[:, 0]`; Julia `samples[:, 1]`.

**F8. `show` prints 6 significant digits and there is no string formatting.** sev 2, strings/IO, 1 summary
(a) The per-cloud summary is one line in ref.py (an f-string with six numbers). In Viv each number
needs its own named local so that `show` labels it (a bare `show sum(x) / len(x)` inside a def
prints the value with no label), which is 13 lines for the six numbers, and the name of the cloud
comes from a separate `narrate` line. Agreement cannot be read from 6 digits, so every comparison
with torch is a computed `*_miss` value plus an `assert_allclose` in the program.
(b) `print("samples: mean ({mean_x}, {mean_y}) ...")` or `show mean_x, mean_y`.
(c) `print(f"... {d.mean():.6g}")`; Julia `@printf`.

**F9. No file input.** sev 2, strings/IO, 1
(a) `ref.py` writes the torch weights and every reference array as Viv source (`ddpm_ref.viv`,
298 KB of literals), read with `#! use ddpm_ref`.
(b) `THETA0 = read_csv("theta0.csv")`. (c) `np.loadtxt`, `torch.load`; Julia `readdlm`.

**F10. No concatenation.** sev 1, missing builtin, 1
(a) `time_features` fills the sine half and the cosine half of each row with index arithmetic in
a scalar double loop (`E[i, k] = sin(angle)`, `E[i, half + k] = cos(angle)`).
(b) `angle = t_column @ freq_row; E = hcat(sin(angle), cos(angle))`.
(c) `torch.cat((emb.sin(), emb.cos()), dim=1)`; Julia `hcat(sin.(a), cos.(a))`.

**F11. No array copy builtin.** sev 1, missing builtin, 1
(a) `theta = THETA0 + 0.0` (plain assignment aliases, so Adam would overwrite the imported
weights `THETA0` in place).
(b) `theta = copy(THETA0)`. (c) `THETA0.clone()`; Julia `copy(THETA0)`.

**F12. No `mean`/`std`, no `pi`.** sev 1, missing builtin, 4 sites
(a) `sum(x) / len(x)`, a `spread` helper, and `PI = 3.141592653589793` at the top (lib/gto.viv
defines its own `GTO_PI` for the same reason).
(b) `mean(x)`, `std(x)`, `pi`. (c) `x.mean()`, `x.std()`, `math.pi`; Julia `mean`, `std`, `pi`.

**F13. `#! use` fails when the program path has no directory.** sev 1, tooling, 1
(a) `viv run ./ddpm.viv`. (b) `viv run ddpm.viv` prints `no module by that name (looked for
lib/params.viv beside the program and under )`. Same as micrograd's F10; still present.

**F14. No `for` with a start value.** sev 1, control flow, 1
(a) step 0 is done before the loop (its checks need it), so the loop is
`for k to TRAIN_STEPS - 1 { ... slot = floor((k + 1) / REPORT) ... }`.
(b) `for step in 1 to TRAIN_STEPS`. (c) `for step in range(1, TRAIN_STEPS)`; Julia `for step in 2:TRAIN_STEPS`.

**F15. No object to hold a model and its schedule.** sev 1, missing data structure, 1
(a) The schedule (`beta`, `alpha`, `alpha_bar`) and the parameter layout (`model` and ten tensor
names) are globals that `q_sample`, `p_sample` and `layer` read. labml's `DenoiseDiffusion` holds
them in `self`. A record would do it in C but records do not run in the interpreter
(reference.astro, "Records").
(b) `record Diffusion { beta: Arr, alpha: Arr, alpha_bar: Arr }` in both engines.
(c) a class; Julia a `struct`.

## Wins

- **Exact agreement with PyTorch, end to end.** The whole run (3000 Adam steps and 1000 sampling
  steps) matches the unmodified labml code in torch float64 to 3e-15, in both engines, and the
  output of the two engines is byte-identical. Only the linspace ulp and rounding order differ.
- **`rand()` is specified exactly** (MINSTD, integer arithmetic in doubles), so Python can replay it
  and the comparison covers training and sampling, not just one step. Torch's generator cannot be
  replayed from Viv, but Viv's can be replayed from anywhere.
- **`#! use lib/params` + `#! use lib/optim`**: ten tensors in one vector, and
  `optim_step(opt, theta, grad, state, model)` is Adam matching `torch.optim.Adam` to 3e-17 after
  one step and 1.2e-15 after 3000.
- **Whole-array expressions read like torch**: `sigmoid`, `swish`, `mse`, and the `p_sample` mean
  `1.0 / pow(alpha[step], 0.5) * (X - eps_coef * eps_theta)` are one line each, the same as
  labml's. A float step index indexes the schedule directly (`alpha_bar[t[i]]`), which replaces
  `gather` and its reshape.
- **The matrix product is fast in the interpreter** (probe/p5.viv), and native C (2.3 s) is faster
  than the PyTorch reference (4.7 s) at this size.
- **Tests carry the evidence**: each comparison with torch is an `assert_allclose` in the program,
  so both engines check agreement themselves.
- **`figure`/`scatter`** give the samples-vs-target plot from inside the program, in both engines.
