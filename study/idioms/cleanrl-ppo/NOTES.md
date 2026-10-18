# CleanRL PPO (CartPole) in Viv

## Source

- https://github.com/vwxyzjn/cleanrl @ `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`,
  file `cleanrl/ppo.py` (shallow clone at `viv-wt/src/cleanrl-ppo`, not copied here).
- Ported: `layer_init`, `Agent` (critic and actor, 4-64-64-1 and 4-64-64-2, tanh),
  `get_action_and_value`, and the training loop of lines 185-297: rollouts over a
  vector of environments, GAE, flattening, shuffled minibatch epochs, the clipped
  policy loss, the clipped value loss, the entropy bonus, `clip_grad_norm_`, Adam
  with eps 1e-5 and a linearly annealed learning rate, and the logged statistics
  (value/policy loss, entropy, old and new approx KL, clipfrac, explained variance).
- The environment is CartPole-v1 written in Viv: `lib/cartpole`'s `cartpole_accel`
  (the same equations as gymnasium's `CartPoleEnv.step`) with gravity set to 9.8,
  an explicit Euler step of 0.02 s, 10 N pushes, termination at |x| > 2.4 or
  |theta| > 12 degrees, truncation at 500 steps, same-step autoreset as
  gymnasium 0.29's `SyncVectorEnv`.

## Files

| file | what |
|---|---|
| `ppo.viv` | the port, the fixed-trajectory GAE check, and 3 tests holding it to the reference's numbers |
| `reference.py` | imports CleanRL's own `Agent` from the clone, steps gymnasium's own `CartPoleEnv`, trains with torch autograd, `torch.optim.Adam` and `clip_grad_norm_`, float64 |
| `reference_output.txt`, `viv_output.txt`, `native_output.txt` | the runs |
| `probe/p1..p8.viv` | the one-line probes behind friction items F3, F6, F9 and the GRAVITY override |

Run: `viv run ./ppo.viv`, `python3 reference.py` (`--digits` adds the 17-digit
values the tests use).

## Sizes

CleanRL's network unchanged (hidden 64, 9,155 parameters) and CleanRL's
hyperparameters except: 32 steps per rollout instead of 128, 40 iterations
(5,120 environment steps instead of 500,000), learning rate 2.5e-3 instead of
2.5e-4 so that the agent learns inside 40 iterations. `num_envs` 4, 4
minibatches of 32, 4 epochs, gamma 0.99, lambda 0.95, clip 0.2, ent 0.01, vf
0.5, max grad norm 0.5: as CleanRL. 640 Adam steps in all.

## Shared randomness

Both programs draw every random number from Viv's Park-Miller generator
(reimplemented in `reference.py`):

| ppo.py | here (both programs) |
|---|---|
| `orthogonal_` on torch's Gaussian | the same algorithm (QR, signs from diag R; torch's `linalg.qr` in the reference, Gram-Schmidt in Viv) on Box-Muller normals |
| `envs.reset(seed=...)`: `np_random.uniform(-0.05, 0.05)` | `-0.05 + 0.1 * rand()` |
| `probs.sample()` | inverse CDF at one uniform |
| `np.random.shuffle(b_inds)` | Fisher-Yates |

Two further departures, both in the reference so that the programs can agree
to rounding: float64 throughout, and the policy sees the float64 state
(gymnasium returns a float32 copy; Viv has no float32, F17).

## Agreement

- **GAE on a fixed trajectory** (6 steps, 3 envs, episodes ending at steps 2, 4
  and 5 in two of them, bootstrap `next_done` = [0, 1, 0]): all 18 advantages
  equal torch's **bit for bit** (`assert_allclose(..., 0.0)`). The Viv `gae` is
  the same operations in the same order, vectorized over environments the way
  ppo.py is.
- **Training**: the printed log (447 lines: 11 statistics for each of 40
  iterations plus the GAE rows) is identical to the reference's at `%g`
  precision except one line, `policy_loss: -0.0237385` vs `-0.0237386`
  (iteration 31, a value on a 6th-digit rounding boundary). At full precision
  (tests):

  | quantity | difference from torch |
  |---|---|
  | iteration 1 and 2: value loss, policy loss, entropy, approx KL, explained variance | at most 7e-15 (rounding) |
  | iteration 40: value loss 33.144 | 2.5e-9 |
  | iteration 40: entropy 0.6159 / policy loss 5.7e-5 / explained variance 0.173 | 4.5e-10 / 8.8e-12 / 1.4e-10 |
  | actor output weights after 640 Adam steps | 2.3e-10 absolute (about 1e-8 relative) |

  The growth from 1e-15 to 1e-10 over 640 steps is rounding differences
  (Adam's update written as `m/(1-b1^t) / (sqrt(v/(1-b2^t)) + eps)` in
  `lib/optim` vs torch's `lerp_`/`addcdiv_`, torch's reductions, Householder vs
  Gram-Schmidt) accumulating; no action sample or termination flipped, so the
  trajectories are identical.
- **Learning curve** (identical in both programs; mean return of the episodes
  that ended in that iteration, episodes in parentheses):
  iterations 1-8: 23 (3), 39.7, 27.7, 45.7, 48, 34.8 (5), 20.7, 46.3;
  11-14: 66 (4), 37, 165, 113; 17: 164.7 (3); 21-22: 216, 156;
  28-29: 300.3 (3), 218; 31-34: 86.5, 65.7, 32, 34; 35: 223; 39-40: 204, 249.
  Iterations with (0) had no episode end: all four poles stayed up through the
  whole 32-step rollout. Returns go from about 25 (random) to 150-300 in 5,120
  steps, with a relapse at iterations 31-34, which is the early part of
  CleanRL's published CartPole curve. Entropy falls from 0.689 (log 2 = 0.693)
  to about 0.6.

## Both engines (and the third)

`VIV=viv-wt/bin/viv tools/both-engines.sh study/idioms/cleanrl-ppo ppo`:
`3 tests pass, 0 failed; identical output`.

| engine | CPU time |
|---|---|
| interpreter (`viv run`) | 22.6 s |
| native C (`cc -O2`) | 0.32 s |
| direct wasm in node (`tools/run-wasm.mjs`) | 0.87 s, output byte-identical to the other two |
| torch float64, 1 thread (reference) | 2.2 s including import |

The direct wasm run needed a `viv.wasm` runtime, which only a cargo build
produces; the one in the `dft-lens` worktree was used (F19). Before renaming
one global (F8) the C did not build.

## Line counts (ported part only; code lines, no blanks, comments or docstrings)

- ppo.py: `layer_init` + `Agent` 29, the loop from `optimizer = ...` to
  `explained_var` 98 (without `writer`/`print`): **127**. Not counted there but
  used: gymnasium's `CartPoleEnv.step`/`reset` (about 35), torch's `orthogonal_`
  (about 15), `Categorical` and autograd.
- ppo.viv: **395** without tests (tests 30 more), of which 63 are lone `}`.
  By section: arguments 36, environment 20, randomness and init 56
  (Box-Muller, Gram-Schmidt `orthogonal`, Fisher-Yates, `arange`), agent 63
  (layout, `affine`, `mlp`, a hand-written `mlp_backward`, `log_softmax`, `rows`,
  `column`), GAE 22 + variance 5, the fixed-trajectory check 19, the training
  loop 193 (of which the loss-and-gradient block is about 70).

About 120 of the difference is what autograd, `Categorical`, `orthogonal_` and
`torch.randperm`-style helpers provide; about 60 is F3-F7, F11 and F12
(indexing, reshaping and branching that NumPy/torch express as one
expression); the rest is braces.

## Friction ledger

Severity: 1 cosmetic, 2 costs lines, 3 blocks or distorts the algorithm.

### F1. No autodiff for array code: the PPO objective is written twice, once as a value and once as its derivative (missing builtin, sev 3, once, ~70 lines)

ppo.py states the loss and calls `loss.backward()`. In Viv the clipped
objective, the clipped value loss and the entropy bonus have to be
differentiated by hand into `dlogits` and `dvalue`, then pushed through a
hand-written `mlp_backward` for each network. Matching torch also meant
reproducing autograd's conventions: `torch.max` of two tensors sends half the
gradient to each side on a tie (which is every sample whose ratio is inside
the clip range), and `clamp` passes the gradient on the closed interval. The
algorithm's central line becomes branch logic.

- Had to (the ugliest part):
  ```
  pg1 = -A * ratio
  pg2 = -A * min(max(ratio, 1.0 - CLIP_COEF), 1.0 + CLIP_COEF)
  policy_loss = policy_loss + max(pg1, pg2) / M
  inside = 0.0
  if ratio >= 1.0 - CLIP_COEF and ratio <= 1.0 + CLIP_COEF {
      inside = 1.0
  }
  d_ratio = -A * inside
  if pg1 > pg2 {
      d_ratio = -A
  }
  if pg1 == pg2 {
      d_ratio = 0.5 * (-A) + 0.5 * (-A * inside)
  }
  d_logp = d_ratio * ratio / M
  ```
  and the same again for the value loss (`lu`, `lc`, `v_inside`, three cases)
  and for the entropy (`dH = -P[i, j] * (newlogp[i, j] + H)`).
- Wanted: `pg_loss = mean(max(-adv * ratio, -adv * clamp(ratio, 1 - c, 1 + c)))`
  and `grad = gradient(loss, theta)`
- Python: `pg_loss = torch.max(pg_loss1, pg_loss2).mean(); loss.backward()`

### F2. One return value (type system, sev 2, 3 sites)

`get_action_and_value` returns `(action, logprob, entropy, value)`; in Viv that
is four separate computations. `mlp` cannot return its hidden activations with
its output, so `mlp_backward` runs the forward pass again (two extra layers per
network per minibatch).

- Had to: `H1 = tanh(affine(theta, net, X))` and `H2 = ...` repeated inside `mlp_backward`
- Wanted: `out, cache = mlp(theta, net, X)`
- Python: `return action, probs.log_prob(action), probs.entropy(), self.critic(x)`

### F3. No broadcasting of a bias row over a batch (missing builtin, sev 2, 4 sites)

`X @ W + b` with `b` 1 by n is "array shape mismatch" (probe/p1.viv). The bias is
added through an outer product with a column of ones, and each bias gradient
(a column sum) is the transposed trick.

- Had to: `X @ W + ones_column(len(X)) @ b`; `param_add(grad, model, net + 5, ones @ dY)`
- Wanted: `X @ W + b`; `sum(dY, 0)`
- Python: `nn.Linear`; autograd

### F4. No reshape / flatten / `view(-1)` (missing builtin, sev 2, 8 sites)

`obs` is stored flat with row `step * NUM_ENVS + e` by hand so that the
`(num_steps, num_envs, 4) -> (-1, 4)` reshape is free; the five
`(num_steps, num_envs)` buffers are flattened with `take(x, arange(n))`; the
critic's `M x 1` output becomes a vector with `column(V)`, the same trick.

- Had to: `setslab(obs, step * NUM_ENVS + e, slab(next_obs, e))`; `b_values = take(values, flat)`
- Wanted: `obs[step] = next_obs`; `b_values = reshape(values, -1)`
- Python: `obs.reshape((-1,) + shape)`, `values.reshape(-1)`, `newvalue.view(-1)`

### F5. No row gather `A[idx]` (missing builtin, sev 2, 1 site + a 10-line helper)

`take` gathers a vector by a vector of indices, which covers `b_logprobs[mb_inds]`
(a win), but picking rows of a matrix needs a matrix of flat indices built in a
double loop.

- Had to: `rows(A, idx)`: `flat[i, k] = idx[i] * OBS + k`, then `take(A, flat)`
- Wanted: `b_obs[mb_inds]` (or `take_rows(A, idx)`)
- Python: `b_obs[mb_inds]`

### F6. No slicing (missing builtin, sev 2, 1 site)

- Had to: `mb_inds = zeros(MINIBATCH_SIZE)` and a loop copying `b_inds[mb * MINIBATCH_SIZE + i]`
- Wanted: `mb_inds = b_inds[start:end]` (probe/p3.viv: "expected `]`, found `:`")
- Python: `b_inds[start:end]`

### F7. No `arange` (missing builtin, sev 1, 3 sites)

`arange(n)` is a 6-line helper (`linspace(0, n-1, n)` would be the one-liner,
but it is not documented to give exact integers, and an index is truncated).

### F8. A global named like a library function: fine in the interpreter, a C compile error (engine gap / tooling, sev 2, once)

CleanRL's variable name `optimizer` collides with `lib/optim`'s `def optimizer`.
The interpreter ran the program; the C failed to build with
`error: redefinition of 'optimizer' as different kind of symbol` and
`called object type 'viv_arr' is not a function`. Renamed to `opt`. The
makemore port hit the same thing (its F6) with `block_size`.

- Wanted: module namespaces (`optim.adam`), or a Viv-level error in both engines
- Python: `import torch.optim as optim; optimizer = optim.Adam(...)`

### F9. Output is `%g` (6 digits), and a line cannot mix text and numbers (strings/IO, sev 2, every output line)

ppo.py prints `global_step=..., episodic_return=...` per episode and logs to
TensorBoard. Viv prints `label: value` at 6 digits, so the log is per iteration
(`show iteration`, `show mean_return`), the reference was changed to print the
same format, and the one real comparison is the tests with 17-digit numbers
pasted in. To learn *how large* the iteration-40 disagreement was I had to write
a throwaway copy of the program that `show`s the difference vectors. The text
diff of the two logs shows one mismatch that is a rounding-boundary artifact,
not a disagreement.

- Wanted: `print("global_step=", global_step, " episodic_return=", r)`, `fmt(x, 17)`
- Python: f-strings

### F10. Records do not run in the interpreter (engine gap, sev 2, ~20 sites)

The agent cannot be `Agent { critic: Mlp, actor: Mlp }`. It is a `lib/params`
flat vector and a layout table, and a layer is named by arithmetic on tensor
ids: `param_get(theta, model, net + 4)` is "the output weight of whichever
network `net` is". `param_get` copies the matrix out on every forward.

- Wanted: `agent.actor.out.W`
- Python: `self.actor = nn.Sequential(...)`, `agent.actor[4].weight`

### F11. No conditional expression; a Bool is not a number (syntax / type system, sev 2, 9 sites)

Each of these is 3-4 lines: the push direction, the sampled action, `onehot`,
`inside`, `v_inside`, `clipped` (the clipfrac count), `mean_return` when no
episode ended, `next_done`, and the tie cases in F1.

- Had to: `force = FORCE_MAG` / `if action == 0 { force = -FORCE_MAG }`
- Wanted: `force = if action == 1 then FORCE_MAG else -FORCE_MAG`; `clipped = clipped + float(abs(ratio - 1) > c)`
- Python: `force = self.force_mag if action == 1 else -self.force_mag`; `(...).float().mean()`

### F12. No elementwise comparison or `where` on arrays (missing builtin, sev 2, the whole minibatch loss)

`max(a, b)` and `min(a, b)` are elementwise on arrays, so the loss *values*
could be vector expressions, but the masks their gradients need (`ratio` inside
the clip range, which side of the `max` won) cannot, so the minibatch loss is a
scalar loop over samples with running sums.

- Wanted: `mask = abs(ratio - 1.0) > CLIP_COEF` as a 1/0 array, `where(pg1 > pg2, -A, ...)`
- Python: `((ratio - 1.0).abs() > args.clip_coef).float().mean()`

### F13. No `randn`, shuffle, categorical draw, orthogonal init or QR (missing builtin, sev 2, 4 sites, 45 lines)

Box-Muller, Fisher-Yates, the inverse-CDF action draw and a Gram-Schmidt
`orthogonal` are written out. Every RL or ML port writes these again (makemore
wrote the first three).

- Wanted: `randn()`, `shuffle(a)`, `choice(p)`, `qr(A)` or `orthogonal(rows, cols)`
- Python: `torch.nn.init.orthogonal_`, `np.random.shuffle`, `Categorical.sample()`

### F14. `for` counts up only (control flow, sev 1, 2 sites)

GAE walks `reversed(range(num_steps))` and Fisher-Yates walks down; both are
`for k to n { t = n - 1 - k ... }`.

- Wanted: `for t from T - 1 down to 0`
- Python: `for t in reversed(range(args.num_steps))`

### F15. No compound assignment on an array element (syntax, sev 1, 5 sites)

- Had to: `ep_return[e] = ep_return[e] + 1.0`, `elapsed[e] = elapsed[e] + 1.0`
- Wanted: `ep_return[e] += 1.0`

### F16. `lib/optim`'s Adam: eps is not a constructor argument (missing builtin, sev 1, once)

`adam(lr, beta1, beta2)` fixes eps at 1e-8; CleanRL uses 1e-5, set by poking
the settings vector: `opt[OPT_EPS] = 1.0e-5`. Setting the annealed rate is the
same kind of line, `opt[OPT_LR] = frac * LEARNING_RATE`, which reads about as
well as `optimizer.param_groups[0]["lr"] = lrnow`.

### F17. `lib/cartpole` is not CartPole-v1, and there is no float32 (type system, sev 1, once each)

`lib/cartpole` has gravity 9.81 and an RK4 step; CartPole-v1 has 9.8 and
explicit Euler. `cartpole_accel` is the same formula as gymnasium's, and
assigning `GRAVITY = 9.8` after `#! use lib/cartpole` changes it in both engines
(probe/p8.viv), so only the 5-line Euler step and the termination test were new.
Rebinding a library's global works but nothing marks it as intended. Gymnasium
hands the agent a float32 observation; Viv has only f64, so the reference was
changed to use the float64 state.

### F18. Undocumented `shape` (tooling, sev 1, 2 sites)

The number of columns is written `len(transpose(G))`; `shape(A)[1]` works but is
not in the reference's builtin tables (makemore F18).

### F19. The third engine needs a cargo-built runtime (tooling, sev 1, once)

`viv ppo.viv -o ppo.wasm` compiles, but `tools/run-wasm.mjs` loads its runtime
from `www/public/viv/viv.wasm`, which only `www/build-wasm.sh` (cargo) makes.
`VIV_WASM=` pointed at another worktree's build ran it byte-identical.

### F20. Interpreter speed sets the problem size (performance, sev 2, once)

22.6 s for 5,120 environment steps. CleanRL's default 500,000 steps would take
about 37 minutes interpreted and about 30 s compiled to C (native is 70 times
faster than the interpreter and 7 times faster than float64 torch at this size).
The rollout length and step count were cut for the interpreter, not for the
algorithm.

Not hit this time: the interpreter's non-short-circuit `and`/`or` (makemore F5).
`fell(s) or elapsed[e] >= MAX_EPISODE_STEPS` has no guarded index.

## Wins

- **GAE** reads almost line for line like ppo.py, vectorized over environments
  with `slab`/`setslab`, and equals torch's to the last bit:
  ```
  delta = slab(rewards, t) + GAMMA * nextvalues * nextnonterminal - slab(values, t)
  lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
  setslab(advantages, t, lastgaelam)
  ```
- **Gradient clipping in one line**, because `lib/params` keeps every parameter
  in one vector:
  `grad = grad * min(1.0, MAX_GRAD_NORM / (sqrt(sum(grad * grad)) + 1.0e-6))`
  (`clip_grad_norm_` is a library call in torch and a loop over tensors inside it).
- `lib/optim`'s Adam plus `lib/params` reproduced `torch.optim.Adam` over 640
  steps and 9,155 parameters to 1e-10; `lib/nn`'s `softmax_fwd` and
  `lib/cartpole`'s `cartpole_accel` dropped in unchanged.
- `take(b_logprobs, mb_inds)` is `b_logprobs[mb_inds]`.
- The forward pass is array code: `tanh(affine(...))`, `(1.0 - H2 * H2)`,
  `returns = advantages + values`, `variance` as two `sum`s.
- Three engines, byte-identical output, three tests passing in each; the
  native build is 70 times faster than the interpreter.
- Test blocks put the reference's 17-digit numbers next to the code that has
  to match them.
