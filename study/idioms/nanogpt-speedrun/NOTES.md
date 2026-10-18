# nanogpt-speedrun: the modded-nanogpt model and optimizer in Viv

**Source.** https://github.com/KellerJordan/modded-nanogpt, `train_gpt.py` at commit
`bc3a0c2d640d0d73dedaef87eae26148d2e32afb` (2026-09-18). Shallow clone in
`viv-wt/src/nanogpt-speedrun` (not copied here).

**Files.**

| file | what |
|---|---|
| `gpt.viv` | the port: model forward and backward, NorMuon + Adam, schedules, training loop; one central-difference test |
| `check.viv` | loads PyTorch's weights and compares loss, every gradient, one optimizer step, and a 31-point training curve |
| `train.viv` | the same training from Viv's own initialization (the file's init recipe), no Python needed |
| `text.viv` | the corpus (the first paragraph of *A Tale of Two Cities*) and its tokenizer |
| `reference.py` | float64 PyTorch version of the ported parts, with autograd; writes `ref.viv` |
| `ref.viv` | generated, 1.1 MB, git-ignored: run `python3 reference.py` before `check.viv` |
| `ref_out.txt` | what `reference.py` printed |
| `SIDE-BY-SIDE.md` | the attention block and the training step in the file, in `reference.py`, and in Viv |

## What was ported, and what was not

The current `train_gpt.py` is 2780 lines plus 2500 lines of Triton kernels, and much of the
model is tricks layered on over a year of records. I kept every piece that is part of the
model's math at any size and that has a meaning with 2 layers:

- vocabulary padded up to a multiple (here 32; the file pads 50257 to 50304);
- token embedding tied to `lm_head` (stored `D x V`, as `CastedLinearT` stores it), with
  `lm_head.grad += embed.grad.T` on Adam steps, `embed = lm_head.T` after them, and the untie at
  the start of the third stage, which copies the Adam state across transposed;
- the smear gate (`x[t] += smear_lambda * sigmoid(W x[t, :12]) * x[t-1]`);
- `x0 = norm(x)` and the x0 injection gates, from the pre MUDD gate network
  (`(gelu(x0 W1^T) W2^T + b2) * scale`, W2 and b2 zero at init, scale learnable);
- attention: `sa_lambdas`-scaled QKV projection, QK RMS-norm per head, half-truncated RoPE with
  the `(1/1024)^linspace` frequencies, the partial key offset on the long-window layer, sliding
  windows (8 and 16 tokens), `qk_dim` narrower than `head_dim` (8 vs 16, as 96 vs 128),
  softmax scale `0.1 sqrt(head_dim / qk_dim)`, value embeddings on layer 1 with the
  `2 sigmoid(W [x[:6], ve[:6]])` per-head gate, `o_w` stored transposed;
- `resid_lambdas` and `post_lambdas` on both branches;
- ReLU-squared MLP, `c_proj` zero-initialized;
- final norm, softcap `23 sigmoid((z + 5) / 7.5)`, cross-entropy summed over tokens;
- NorMuon: Nesterov momentum in lerp form, Polar Express (the five coefficient triples, the
  `1.02` safety factor, the tall/wide branches), per-head-pair groups for the QK bank, the NorMuon
  variance reduction, cautious weight decay, the `sqrt(rows / cols)` shape multiplier, and 2x lr
  on `c_proj`;
- Adam with bias correction and cautious weight decay, per-parameter betas, `lr_mul` and
  `wd_mul` from the file's `param_table`, stepped on odd steps only with gradients accumulating
  over the even step;
- the lr schedule (three stages with `lr_mul` 1, 1.52, 1.73, cooldown over the last 60% to 0.15)
  and the Muon momentum warmup 0.85 to 0.95 and cooldown, scaled from 1290 steps to 30.

Left out, and why: MUDD residual mixing and the post MUDD gate (they wire layers 3, 7 and 9
together, and exist only at depth), the layer-6 attention skip (same), bigram hash embeddings
(a 377k-row table and a sparse gradient path; possible, but a second model), XSA and DC
attention and attention output gates (per-layer features on layers 1, 3, 4, 7, 10), paired
heads, fp8/bf16 and the mantissa tracking that exists because of bf16 (the port is float64
throughout, which is what makes a 1e-15 comparison possible), MTP and prefix-token losses,
the YaRN window extension, the canonical-token validation mask, and all distributed and
sharding code.

## Sizes

vocab 32 (25 characters used), context 16, width 32, 2 heads of `head_dim` 16 with `qk_dim` 8,
MLP width 128, gate width 16, 2 layers, 4 sequences per step, 30 steps. 26,196 parameters.

## Evidence it is right

`check.viv` against `reference.py` (float64 PyTorch, autograd). The gradient check uses a
perturbed weight set in which every weight is nonzero, because at the file's initialization
`c_proj`, the gate `W2` and the smear gate are zero and several gradients vanish.

| quantity | Viv | PyTorch | agreement |
|---|---|---|---|
| summed loss over 4x16 tokens, perturbed weights | 221.91 | 2.219101156505677e+02 | within 1e-10 relative (asserted) |
| gradient norm of each of the 22 tensors | | | worst relative gap **1.9e-15** |
| `qk1` and `embed` gradients, entry by entry | | | max gap / max entry < 1e-10 (asserted) |
| one optimizer step (step 1: NorMuon and Adam) | | | worst relative gap in the per-tensor step norms **3.5e-15**; `fc0` and `lm_head` after the step match entry by entry to 1e-10 |
| loss per token before each of 30 steps and after the last, from the file's init | 3.46681 → 2.8146 | 3.466809424659 → 2.814602329869 | max absolute gap **1.3e-15** |

`gpt.viv`'s own test checks two random entries of every tensor's gradient against central
differences of the loss (worst 8.8e-7 relative, the truncation error of `h = 1e-5`; 1.9e-6
before lib/nn reduced along an axis in NumPy's order, 2026-10-02).

The training run crosses every schedule event: the stage changes at steps 10 and 20, the
cooldown from step 12, the momentum warmup and cooldown, the untie at step 21, and 15 Adam
steps with accumulated gradients. It starts at `log 32 = 3.4657` and ends near 2.6 per token
(the per-step numbers are on different batches, so the curve is noisy).

`train.viv`, from Viv's own initialization (Box-Muller normals, `srand(2026)`), goes from
3.471 to 2.654 at step 25. Its normals are `lib/random`'s `randn`, `sqrt(-2 log u1) cos(2 pi u2)`;
the port first wrote its own, `sqrt(-2 log(1 - u1)) cos(2 pi u2)`, which draws a different
normal from the same two uniforms and went from 3.472 to 2.660. `train.viv` is compared with
no reference (its test checks that it starts at chance and learns), so only the trajectory
moved; `check.viv`, which is held to `reference.py`, draws no normals.

## Both engines

`tools/both-engines.sh study/idioms/nanogpt-speedrun gpt check train` with the shared binary:

```
gpt                                 1 tests pass, 0 failed; identical output
check                               6 tests pass, 0 failed; identical output
train                               1 tests pass, 0 failed; identical output
```

| program | interpreter | native (`cc -O2`) |
|---|---|---|
| `check.viv` | 18.5 s | 0.16 s (plus 1.6 s to compile the C) |
| `train.viv` | 19 s | about the same as check |
| `gpt.viv` (its test) | 3.6 s | |

The direct wasm backend compiles `check.viv` (579 KB module), but it was not run: the worktree has
no `viv.wasm` runtime for `tools/run-wasm.mjs`, and building one needs cargo, which this study was
not to run.

## Line counts (the ported part only; code lines, no comments or blank lines)

| part | `train_gpt.py` | `reference.py` (plain PyTorch) | Viv |
|---|---|---|---|
| model forward | ~200 across `GPT.forward`, `CausalSelfAttention.forward`, `Yarn`, plus a ~110-line Triton QK-norm/RoPE kernel pair | 57 | ~215 |
| backward | 0 (autograd; the Triton kernels bring their own backward kernels) | 0 | ~196 (150 in `sequence`, 34 `qk_norm_rope_bwd`, 12 small adjoints) |
| parameters, layout, config | ~120 (`__init__`s, `param_table`) | 48 | 57 (+ 35 for the init in `train.viv`) |
| optimizer, schedules, loop | ~450 (`NorMuonAndAdam`, `polar_express`, schedules, `TrainingManager`; most of it comms) | 143 | 238 |
| total | | ~284 | ~724 |

Against the plain PyTorch version: the forward is 3.8 times as long, the backward is all
extra, and the optimizer is 1.7 times as long. Against the file itself, the Viv optimizer is
shorter, because the file's is mostly distributed-communication code.

## Friction ledger

Severity: 1 cosmetic, 2 costs lines, 3 blocks or distorts the algorithm.

### F1. No autograd: the backward pass is written by hand (missing builtin, severity 3, once, for 14 operations)

(a) Had to write: 196 lines of adjoints (softcapped cross-entropy, norm, head, residual scalars,
ReLU^2 MLP, attention per head, softmax, QK-norm + RoPE with the key offset, value-embedding
gate, QKV projection with `sa_lambdas`, x0 gate network with GELU, smear gate, two embedding
scatters), plus a central-difference test and a PyTorch comparison to know they are right.
The ugliest part is the key offset, whose gradient crosses tokens:

```viv
shifted = offset > 0.0 and t > 0 and h >= H and d >= ROT
if not shifted {
    dN[t, c + d] = dN[t, c + d] + F1[t, d] * dY[t, c + d]
    dN[t, c + e] = dN[t, c + e] + F2[t, d] * dY[t, c + d]
}
if shifted {
    dN[t - 1, c + d] = dN[t - 1, c + d] + dY[t, c + d]
}
```

(b) Wished: `loss, grad = value_and_grad(batch_loss)(theta, tokens, step)` (reverse mode over
array code).
(c) Python: `loss.backward()`; Julia: `Zygote.withgradient(θ -> batch_loss(θ, tokens, step), θ)`.

It distorts the program's shape, not only its length: forward and backward share one 320-line
function, `sequence`, so that the forward's activations are still in scope (see F2), and every
new trick in the model costs its adjoint too. Adding the value-embedding gate was 17 forward
lines and 23 backward lines.

### F2. Records do not run in the interpreter, so a forward pass cannot hand its activations to a backward pass (engine gap, severity 3, once, 18 arrays)

(a) Had to write: 18 rank-3 and rank-4 locals allocated at the top of `sequence`, written with
`setslab` in the layer loop and read back with `slab` in the backward loop:

```viv
XIN = zeros(L, T, D)
XN = zeros(L, T, D)
RN = zeros(L, T)
QKR = zeros(L, T, 2 * H * QK)
...                          # 14 more
PATT = zeros(L, H, T, T)
```

(b) Wished: `record LayerCache { xin: Arr, xn: Arr, r: Arr, qk: Arr, ... }` and
`cache = layer_forward(theta, l, x)` / `dx = layer_backward(theta, grad, l, cache, dx)`, in both
engines (records work in C; `record` in the interpreter stops with "use the Compile to C tab").
(c) Python: a tuple or dataclass; `ctx.save_for_backward(...)`; Julia: a NamedTuple.

### F3. No views, slices or concatenation: heads and feature ranges are index arithmetic (missing data structure, severity 2, about 30 index expressions and 6 copy loops)

(a) Had to write: `y[j, (H + h) * QK + d]` for "key head h, lane d"; `v[t, h * HD + d]`; the
RoPE pair partner `e = d + 1 - 2 * mod(d, 2)`; a loop to build `cat(xn[:, :6], ve[:, :6])`;
a loop to read `x[t, :12]` for the smear gate.
(b) Wished: `qk = reshape(raw, T, 2 * H, QK)`, `q = qk[:, h, :]`, `x[:, 0:12]`, `hcat(A, B)`.
(c) Python: `qk.view(T, 2 * H, QK)`, `torch.cat([xn[:, :6], ve[:, :6]], -1)`,
`n.view(T, 2*H, QK//2, 2).flip(-1)`; Julia: `reshape(raw, QK, 2H, T)`, `[xn[:, 1:6] ve[:, 1:6]]`.

### F4. No broadcasting beyond scalars: row and column broadcasts are loops (missing builtin, severity 2, 5 places)

(a) Had to write, for `x0 * gate[:, l:l+1]`:

```viv
for t to T {
    for d to D {
        xm[t, d] = xm[t, d] + X0[t, d] * GATE[t, l]
    }
}
```

and the same for `+ b2`, `veg[:, :, None] * ve`, and in the backward the matching reductions
(`dGATE[t, l] += dxm[t, d] * X0[t, d]`, `dB2[0, l] += dGRAW[t, l]`).
(b) Wished: `xm = xm + X0 .* col(GATE, l)` and `sum(dGRAW, axis 0)`.
(c) Python: `x0 * gate[:, l:l+1]`, `grad.sum(0)`; Julia: `X0 .* GATE[:, l]`, `sum(dG, dims=1)`.

### F5. No `+=` on an array element (syntax, severity 2, about 25 lines)

(a) Had to write: `dy[j, (H + h) * QK + d] = dy[j, (H + h) * QK + d] + g * y[i, h * QK + d]`.
Every accumulation in a backward pass repeats its left side, and a long index repeated is where
a typo hides.
(b) Wished: `dy[j, (H + h) * QK + d] += g * y[i, h * QK + d]`.
(c) Python/Julia: `+=`.

### F6. No tuple returns: several results come back through out-parameters (type system, severity 2, 5 functions)

(a) Had to write: `y = qk_norm_rope(raw, KEY_OFFSET[l], nq, rq)` with `nq` and `rq` allocated
by the caller first; `norm_rows(x, r)`; `relu2_bwd(pre, T, M, dY, dpre)`;
`cross_entropy_bwd(LOGITS, tgt, T, V, dlogits)`; `softmax_bwd(Pr, T, T, dP, dS)`.
(b) Wished: `y, n, r = qk_norm_rope(raw, offset)`.
(c) Python: `return y, n, r`; Julia: the same.

### F7. No shape query: the number of columns is `len(transpose(X))` (missing builtin, severity 1, 8 places)

(a) Had to write: `w = len(transpose(X))`, which builds a transposed copy to read one number.
The lib functions sidestep it by taking the sizes as arguments (`rmsnorm_bwd(X, T, D, r, dY, dX)`,
`softmax_fwd(S, T, T)`), which is its own friction: every call restates shapes the array knows.
(b) Wished: `rows, cols = size(X)` or `cols(X)`.
(c) Python: `X.shape`; Julia: `size(X)`.

### F8. One def cannot take a number at one call and an array at another in C (engine gap / type system, severity 1, once)

(a) Had to write: `sigmoid(x)` for the scalar smear gate and `sigmoids(X: Arr)` for the gates
and the softcap, with the same body. The interpreter ran the single version; the C backend
typed `sigmoid` by its first call and rejected the second: "argument 1 of `sigmoid` should be
a number but is a array".
(b) Wished: one `def sigmoid(x) { return 1.0 / (1.0 + exp(-x)) }`, specialized per call, as
`exp` itself is.
(c) Python: `torch.sigmoid` on anything; Julia: `sigmoid.(x)` from one scalar definition.

### F9. No record or dict for per-parameter optimizer settings (missing data structure, severity 2, one table, 7 reads)

(a) Had to write: a 22 x 5 numeric table whose columns mean optimizer, beta1, beta2, lr_mul,
wd_mul, read as `OPT[k, 3]`; and the optimizer's state as six loose arguments threaded through
every call (`optimizer_step(theta, grad, first, second, steps, split, step)`).
(b) Wished: `record AdamCfg { beta1: Num, beta2: Num, lr_mul: Num, wd_mul: Num }` in a
table keyed by tensor, and one `state` value.
(c) Python: `param_table = {"lm_head": {"optim": "adam", "adam_betas": [0.5, 0.95], ...}}`.

### F10. A function cannot update a scalar its caller holds (type system, severity 1, once)

(a) Had to write: `split = zeros(1)` and `split[0] = 1.0` for the "embed is untied" flag.
(b) Wished: `split` in a state record, or a returned value.
(c) Python: `self.split_embed = True`.

### F11. Library constants cannot be parameters (missing builtin, severity 1, 2 places)

(a) `lib/nn`'s `rmsnorm_fwd` hardcodes `NN_EPS = 1e-8`; the speedrun's kernel uses the float32
epsilon, 1.19e-7, and at 1e-15 agreement that matters. I wrote `norm_rows` (13 lines) and kept
`lib/nn`'s `rmsnorm_bwd`, which does not need the epsilon. `lib/optim`'s Muon is plain
Newton-Schulz with one optimizer per layout, so NorMuon + Adam per tensor had to be written
anyway. And `gpt.viv`'s own `STEPS` and `STAGE_END` are module globals: `train.viv` cannot ask
for 90 steps without editing `gpt.viv`.
(b) Wished: `rmsnorm_fwd(X, r, eps: 1.19e-7)` with a default; a module parameter.
(c) Python: `F.rms_norm(x, shape, eps=...)`; a `Hyperparameters` dataclass.

### F12. No file input and no string values: the reference weights are Viv source (strings/IO, severity 2, once)

(a) Had to write: `reference.py` emits a 1.1 MB `ref.viv` in which each weight set is a
function that returns one of 22 matrix literals by index, because an array of arrays of
different shapes is not a value:

```viv
def ref_pert(k) {
    if k == 0 {
        return [[...], [...], ...]
    }
    if k == 1 {
    ...
```

(b) Wished: `theta = load("weights.npz")` or `read_matrix("w_lm.txt")`, or a list of arrays.
(c) Python: `torch.load`, `np.loadtxt`; Julia: `readdlm`.

### F13. No `randn` (missing builtin, severity 1, once)

(a) Had to write: Box-Muller, `sqrt(-2.0 * log(1.0 - rand())) * cos(2.0 * 3.14159... * rand())`.
(b) Wished: `randn()`. (c) Python: `torch.randn`; Julia: `randn()`.

### F14. Numeric flags instead of booleans in arrays (type system, severity 1, 3 arrays)

(a) Had to write: `KEY_OFFSET = [0.0, 1.0]` and `if KEY_OFFSET[l] > 0.0`.
(b) Wished: `KEY_OFFSET = [false, true]`.
(c) Python: `key_offset = [b == ws_long for b in bm_sizes]`.

### F15. Errors inside a used module point into the spliced text, not the module (tooling/error message, severity 2, hit once)

(a) The `sigmoid` type error in `gpt.viv` was reported as `./check.viv:1902:34`, a line number
in the concatenation of `check.viv`, `gpt.viv`, `ref.viv` and three libraries. Finding it meant
searching for the quoted source line.
(b) Wished: `gpt.viv:331:34`, where the call was.

### F16. `viv run check.viv` from the program's own directory cannot find `lib/` (tooling, severity 1, hit once)

(a) With a bare file name the program's directory is the empty path, the search for
`Cargo.toml` never starts, and the error reads "looked for lib/nn.viv beside the program and
under " (the root is blank). `viv run ./check.viv` works.
(b) Wished: the bare name resolves like `./check.viv`.

### F17. `show` prints six significant digits (strings/IO, severity 1, throughout)

(a) The loss prints as `221.91`; a 1e-12 agreement cannot be shown, only asserted. I compared
by asserting against numbers exported from PyTorch, which is the better check anyway, but a
reader cannot see the digits.
(b) Wished: `show loss %.15e` or `print(x, 15)`.
(c) Python: `f"{loss:.15e}"`.

### F18. The interpreter is 115 times slower than native (performance, severity 1, throughout)

`check.viv`: 18.5 s interpreted, 0.16 s native. Within this study's budget, but it set the sizes
(4 sequences of 16, 30 steps) and a real char-level run would have to go native.

## Wins

- **It matches autograd to rounding.** Every gradient within 2e-15 relative, the optimizer step
  within 4e-15, 31 training losses within 1.3e-15, and the interpreter and C print the same bytes.
- **Matrix algebra reads like NumPy.** `@`, `transpose`, elementwise `exp`, `erf`, `max`, and
  scalar broadcasting from either side give lines like
  `U = (XF @ LMW + 5.0) / 7.5` and `dZ = (T * 23.0 / 7.5) * dlogits * SIG * (1.0 - SIG)`.
- **`lib/polar` made Polar Express 12 lines.** `ns_step` (`a X + b (X X^T) X + c (X X^T)^2 X`)
  and `frob` are exactly the pieces it needs; the per-step coefficients are a 5 x 3 literal and a
  loop. The file needs 80 lines and three Triton kernels (for speed, not for math).
- **`lib/nn` supplied five adjoints**: `softmax_fwd/bwd` (the masked scores need nothing special,
  a masked weight is exactly zero), `relu2_fwd/bwd`, `cross_entropy_fwd/bwd`, `rmsnorm_bwd`.
- **`lib/params` made the tied embedding and its untie one-liners.**
  `param_add(grad, model, LM, transpose(param_get(grad, model, EMB)))` is `transpose_add`, and
  copying Adam's state across at the split is two `param_set`s. Declaring a tensor returns the
  name the program calls it by.
- **Adam-on-odd-steps with accumulated gradients cost nothing**: gradients are added into
  `grad` with `param_add`, and the step clears only the tensors that stepped.
- **`chars("...")`** made the corpus and its tokenizer 15 lines, in both engines.
- **Functions are values**: `random_matrix(rows, cols, scale, randn)` takes the distribution
  and calls it with `apply(dist)`, and the C backend specializes it per call.
- **`test` blocks** made the evidence part of the program: `check.viv` fails if any number
  drifts from PyTorch's, and `gpt.viv` checks its own backward pass by central differences.
