# Side by side: modded-nanogpt and the Viv port

Reference: `train_gpt.py` in [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
at commit `bc3a0c2d640d0d73dedaef87eae26148d2e32afb`. Each quote from it is a short excerpt, with
its line numbers. Between the two is `reference.py`, the minimal float64 PyTorch version written
for this study, which is what the Viv port is checked against.

## 1. The attention block

### train_gpt.py (`CausalSelfAttention.forward`, lines 1146-1154, 1169-1171, 1195-1196)

```python
qkv_weight = sa_lambdas[0] * torch.cat((qk_w, v_w)).type_as(x)
qkv = F.linear(x, qkv_weight)
qk, v = qkv.split((self.qk_proj_dim, self.hdim), dim=-1)
qk = qk.view(B, T, 2 * self.num_heads, self.qk_dim)
v = v.view(B, T, self.num_heads, self.head_dim)
q, k = QKNormRoPEPad(qk[0], yarn.factor1[:T], yarn.factor2[:T], self.num_heads,
                     self.attn_qk_dim, self.paired, key_offset)
...
y = flash_attn_interface.flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens_q=seqlens, ...,
        causal=True, softmax_scale=yarn.attn_scale, window_size=(bm_size, 0))
...
out_weight_t = sa_lambdas[1] * o_w.type_as(y)
y = y @ out_weight_t
```

QK-norm and RoPE are a Triton kernel (`_qk_norm_rope_pad_forward_kernel`, about 110 lines
with its backward kernel beside it); attention is FlashAttention 3. Autograd supplies the backward.

### reference.py (the same, in plain PyTorch)

```python
xn = norm(x)
qk = (xn @ (sa[0] * P[f"qk{l}"]).T).view(T, 2 * H, QK)
v = (xn @ (sa[0] * P[f"v{l}"]).T).view(T, H, HD)
q, k = qk_norm_rope(qk, KEY_OFFSET[l])              # 9 lines, below
if VE_LAYERS[l]:
    gate_in = torch.cat([xn[:, :6], ve[:, :6]], dim=-1)
    veg = 2 * torch.sigmoid(gate_in @ P["ve_gate"].T)
    v = v + veg[:, :, None] * ve.view(T, H, HD)
s = torch.einsum("ihd,jhd->hij", q, k) * ATTN_SCALE
s = s.masked_fill(~MASKS[l], float("-inf"))
p = torch.softmax(s, dim=-1)
y = torch.einsum("hij,jhd->ihd", p, v).reshape(T, D)
out = y @ (sa[1] * P[f"o{l}"])
```

### gpt.viv (forward, inside `sequence`)

```viv
xn = norm_rows(x, r)
raw = xn @ transpose(sa0 * param_get(theta, model, WQK[l]))
y = qk_norm_rope(raw, KEY_OFFSET[l], nq, rq)          # 33 lines, with index arithmetic for the heads
v = xn @ transpose(sa0 * param_get(theta, model, WV[l]))
if HAS_VE[l] > 0.0 {
    for t to T {                                      # torch.cat([xn[:, :6], ve[:, :6]])
        for c to 6 {
            GIN[t, c] = xn[t, c]
            GIN[t, 6 + c] = VEROWS[t, c]
        }
    }
    VEGS = 2.0 * sigmoids(GIN @ transpose(VEGW))
    for t to T {                                      # veg[:, :, None] * ve.view(T, H, HD)
        for h to H {
            for d to HD {
                v[t, h * HD + d] = v[t, h * HD + d] + VEGS[t, h] * VEROWS[t, h * HD + d]
            }
        }
    }
}
ya = zeros(T, D)
for h to H {                                          # einsum, mask, softmax, einsum
    S = zeros(T, T)
    for i to T {
        for j to T {
            S[i, j] = MASKED
            if j <= i and j >= i - WINDOW[l] {
                total = 0.0
                for d to QK {
                    total = total + y[i, h * QK + d] * y[j, (H + h) * QK + d]
                }
                S[i, j] = ATTN_SCALE * total
            }
        }
    }
    Pr = softmax_rows(S)                               # lib/nn
    for i to T {
        for j to T {
            PATT[l, h, i, j] = Pr[i, j]                # kept for the backward pass
            for d to HD {
                ya[i, h * HD + d] = ya[i, h * HD + d] + Pr[i, j] * v[j, h * HD + d]
            }
        }
    }
}
out = ya @ (sa1 * param_get(theta, model, WO[l]))
```

The matrix products read the same in all three (`xn @ transpose(sa0 * W)` against
`F.linear(x, sa * W)`). What costs lines in Viv is everything between them: the head dimension
is `h * HD + d` instead of a `.view`, a column slice and a concatenation are a loop, a
`[:, :, None]` broadcast is a loop, and `einsum` is three nested loops.

### gpt.viv (the same block's backward pass, which PyTorch does not need written)

```viv
dy = zeros(T, 2 * H * QK)
dv = zeros(T, D)
for h to H {
    Pr = zeros(T, T)
    dP = zeros(T, T)
    for i to T {
        for j to T {
            Pr[i, j] = PATT[l, h, i, j]
            total = 0.0
            for d to HD {
                total = total + dya[i, h * HD + d] * v[j, h * HD + d]
                dv[j, h * HD + d] = dv[j, h * HD + d] + Pr[i, j] * dya[i, h * HD + d]
            }
            dP[i, j] = total
        }
    }
    dS = softmax_bwd(Pr, dP)
    for i to T {
        for j to T {
            g = ATTN_SCALE * dS[i, j]
            for d to QK {
                dy[i, h * QK + d] = dy[i, h * QK + d] + g * y[j, (H + h) * QK + d]
                dy[j, (H + h) * QK + d] = dy[j, (H + h) * QK + d] + g * y[i, h * QK + d]
            }
        }
    }
}
```

Plus 23 lines for the value-embedding gate's backward, 34 for `qk_norm_rope_bwd` (including
the key offset, which sends a key's lanes' gradient to the token before it) and 8 for the
QKV projection and `sa_lambdas`.

## 2. The training step

### train_gpt.py (lines 2753-2760, and `step_optimizers`, lines 2509-2525)

```python
for idx in range(grad_accum_steps):
    inputs, targets, cum_seqlens, bigram_inputs, bigram_cpu = train_loader.send(...)
    loss = model(inputs, targets, cum_seqlens, bigram_inputs,
                 training_manager.get_forward_args()).sum() * grad_scale
    loss.backward()
training_manager.step_optimizers(step)

def step_optimizers(self, step: int):
    step_lr = training_schedule.get_lr(step)
    muon_momentum = get_muon_momentum(step)
    do_adam = self._is_adam_step(step)
    for param, p_cfg in self.optimizer.param_cfgs.items():
        p_cfg.lr = p_cfg.initial_lr * step_lr
        if p_cfg.optim == "normuon":
            p_cfg.momentum = muon_momentum
    self.optimizer.step(do_adam=do_adam)
    if step == self.split_step:
        self.optimizer.copy_lm_state_to_embed()
```

The NorMuon update itself (`_normuon_update`, lines 907-916):

```python
v_chunk = polar_express(grad_chunk, p_state["momentum_buffer"], self._momentum_t, ...)
red_dim = -1 if chunk_shape[-2] >= chunk_shape[-1] else -2
v_chunk = NorMuonAndAdam._apply_normuon_variance_reduction(
    v_chunk, p_state["second_momentum_buffer"], p_cfg.beta2, red_dim)
```

and Polar Express's loop (wide branch, lines 268-278; the comments are the file's):

```python
for a, b, c in polar_express_coeffs:
    XXT(X, out=A)  # A = X @ X.mT
    ba_plus_cAA(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A
    aX_plus_BX(X, B, X, beta=a, out=C)  # C = a * X + B @ X
    X, C = C, X
```

### gpt.viv

```viv
def train(theta: Arr, tokens: Arr, losses: Arr) {
    grad = param_zeros(model)
    first = param_zeros(model)
    second = param_zeros(model)
    steps = zeros(param_tensors(model))
    split = zeros(1)
    for step to STEPS {
        losses[step] = batch_loss(theta, grad, tokens, step, true) / (B * T)
        optimizer_step(theta, grad, first, second, steps, split, step)
    }
    losses[STEPS] = batch_loss(theta, grad, tokens, STEPS, false) / (B * T)
}

def optimizer_step(theta: Arr, grad: Arr, first: Arr, second: Arr, steps: Arr, split: Arr, step) {
    lr_sched = lr_schedule(step)
    mu = muon_momentum(step)
    do_adam = mod(step, 2) == 1
    tied = split[0] == 0.0
    if do_adam and tied {
        param_add(grad, model, LM, transpose(param_get(grad, model, EMB)))
    }
    for k to param_tensors(model) {
        if OPT[k, 0] == NORMUON {
            normuon_step(theta, grad, first, second, k, lr_sched, mu)
        }
        else {
            if do_adam and not (k == EMB and tied) {
                adam_step(theta, grad, first, second, steps, k, lr_sched)
            }
        }
    }
    ...   # re-tie, clear the gradients that stepped, untie at SPLIT_STEP: 14 more lines
}

def polar_express(Gm: Arr) {
    rows = len(Gm)
    cols = len(transpose(Gm))
    X = Gm / (frob(Gm) * 1.02 + 1.0e-6)
    if rows > cols {
        X = transpose(X)
    }
    for k to 5 {
        X = ns_step(X, POLAR_EXPRESS[k, 0], POLAR_EXPRESS[k, 1], POLAR_EXPRESS[k, 2])   # lib/polar
    }
    if rows > cols {
        return transpose(X)
    }
    return X
}
```

The loop and the optimizer's control flow are about as long as the file's, and Polar Express is
shorter, because `lib/polar` already has the quintic step. Two things are different in kind:

- `loss.backward()` has no counterpart. `batch_loss(..., true)` runs a forward pass and then the
  hand-written backward pass that lives in the same function (`sequence`, 150 lines of backward).
- The optimizer's state is six loose arguments (`theta, grad, first, second, steps, split`) where
  the file has a `param_cfgs` dict of dataclasses and a `param_states` dict of dicts, and the
  per-parameter hyperparameters are a numeric table read by column number (`OPT[k, 3]` is
  `lr_mul`) where the file writes `p_cfg.lr_mul`. `split` is a one-element array because a
  function cannot set a scalar that its caller sees.
