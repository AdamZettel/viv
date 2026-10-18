---
title: The same GPT in PyTorch, against Viv
order: 2
output: identical
---
- The Python column is the model of [tiny_gpt](/compare/ml/tiny_gpt) in
  PyTorch: `nn.Module` subclasses, `nn.Linear` and `nn.Embedding` layers,
  `F.scaled_dot_product_attention(..., is_causal=True)`, `F.cross_entropy`,
  `torch.optim.Adam` with a `LambdaLR` schedule, and Muon as an 11-line
  `torch.optim.Optimizer` subclass. There is no backward code: `loss.backward()`
  fills every `p.grad`, which is why its loss and backward regions hold only a
  comment. It is the reference for what the Viv program would look like with
  autograd, arrays of modules and batched tensors. It needs torch (run with
  2.2.2, in float64 on one thread).
- The Viv column imports what the torch program imports: `#! use lib/nn`
  for `attention` (one causal head, as `F.scaled_dot_product_attention(...,
  is_causal=True)`), `softmax_rows`, `rmsnorm`, `relu2`, `cross_entropy` and
  `embedding` (the row lookup of `nn.Embedding`); `#! use lib/optim` for
  `adam_update` and `muon_update`, which update one matrix and its memories in
  place with the settings in an `AdamSettings` or `MuonSettings` record; and
  `#! use lib/random` for `randn_matrix` and `choice`. It is the program
  `lib/gpt/story_gpt.viv` runs with `lib/gpt/gpt.viv` copied in, and prints
  the same bytes as the written-out Viv of the NumPy pair
  ([tiny_gpt](/compare/ml/tiny_gpt)): all 30 losses and every trained weight
  to 17 digits, and the sample, in the interpreter, in C and in wasm. Its
  model is the same tree as the torch modules, as records: `Gpt { tok, pos,
  blocks: [Block], unembed }`, `Block { attn, mlp }`, `Attention { heads:
  [Head], Wo }`, `Head { Wq, Wk, Wv }`; its gradient is
  `value_and_grad(sequence_loss, model, ...)`, a `Gpt`; and `optimizer_step`
  is `for_leaves(muon_update, model.blocks, g.blocks, m.blocks, lr_scale,
  MUON_OPT)` and three `adam_update` calls, where torch calls `o.step()` on
  each optimizer.
- The outputs are the same text. Printed with 17 digits, the 30 losses differ
  from Viv's by at most 6.7e-16. The weights are copied from the same
  MINSTD draws (`init_from` fills each layer in the order `lib/gpt` lays them
  out, transposed where `nn.Linear` stores `(out, in)`), and the samples use
  the same draws. Muon's Newton-Schulz step returns the matrix unchanged when
  its norm is zero, as `lib/polar` does: at the first step the output map is
  zero, so every block matrix has a zero gradient, and dividing by the norm
  would give NaN.
- torch 2.2 has no `F.rms_norm` and no Muon; both are written out here, in 2
  and 11 lines (plus the 9 of `newton_schulz`). Viv imports both from its
  libraries, so the optimizer region is shorter in Viv (8 lines against 20).
  The torch program writes `Minstd` (12 lines) only to draw the weights and
  the samples Viv draws; Viv's random draws come from `lib/random`.

### Measured

Lines are code lines: blank lines, comments and docstrings are not counted;
characters are those lines' characters. A lookup is a read of a size from the
configuration record (`cfg.<field>`, `c.<field>`); torch's `self.q(x)` is not
counted, and neither is Viv's `h.Wq`. `study/compare/measure.py` counts them.

%% measured
| region | Python lines | Viv lines | Viv / Python | Python lookups | Viv lookups |
|---|---|---|---|---|---|
| imports | 8 | 3 | 0.4 | 0 | 0 |
| config | 15 | 15 | 1.0 | 0 | 0 |
| random draws | 12 | 0 | 0.0 | 0 | 0 |
| parameters and init | 8 | 25 | 3.1 | 0 | 19 |
| embedding | 0 | 0 |  | 0 | 0 |
| attention | 14 | 9 | 0.6 | 6 | 0 |
| block | 11 | 5 | 0.5 | 4 | 0 |
| forward | 13 | 8 | 0.6 | 7 | 0 |
| loss | 0 | 3 |  | 0 | 0 |
| backward | 0 | 0 |  | 0 | 0 |
| optimizer | 20 | 8 | 0.4 | 0 | 0 |
| training loop | 21 | 18 | 0.9 | 4 | 4 |
| sampling | 7 | 10 | 1.4 | 1 | 0 |
| text as tokens | 4 | 4 | 1.0 | 0 | 0 |
| run | 8 | 12 | 1.5 | 0 | 0 |
| checks | 4 | 12 | 3.0 | 0 | 0 |
| **all** | **145** | **132** | **0.9** | **22** | **23** |

| characters of code | Python | Viv | Viv / Python |
|---|---|---|---|
| imports | 188 | 189 | 1.0 |
| config | 307 | 362 | 1.2 |
| random draws | 496 | 0 | 0.0 |
| parameters and init | 318 | 937 | 2.9 |
| embedding | 0 | 0 |  |
| attention | 552 | 231 | 0.4 |
| block | 385 | 155 | 0.4 |
| forward | 440 | 215 | 0.5 |
| loss | 0 | 86 |  |
| backward | 0 | 0 |  |
| optimizer | 714 | 400 | 0.6 |
| training loop | 996 | 627 | 0.6 |
| sampling | 227 | 287 | 1.3 |
| text as tokens | 163 | 159 | 1.0 |
| run | 333 | 398 | 1.2 |
| checks | 176 | 469 | 2.7 |
| **all** | **5295** | **4515** | **0.9** |

| the forward and backward defs | Python | Viv |
|---|---|---|
| defs | 5 | 4 |
| parameters, all of them together | 5 | 9 |
| output parameters (arrays the def itself writes results into) | 0 | 0 |
| most in one def | 1 (`forward`) | 3 (`sequence_loss`) |
%% end

### Where the Viv is still longer

The code follows the house style: a step of one statement
is one line (`step mlp: X = X + relu2(rmsnorm(X) @ b.mlp.up) @ b.mlp.down`),
an array of records is built by a comprehension of constructor calls
(`[Block(c) for l to c.layers]`), and no line is wider than 100 columns, so a record that does not
fit puts one field per line.

- **parameters and init** (25 against 8): the torch layers make and size
  their own weights in their constructors (`nn.Linear(cfg.width, size)`, in
  the attention, block and forward regions); the Viv records only declare
  the matrices, and the constructor defs `Head`, `Block` and `Gpt` build
  them (`model = Gpt(cfg)`).
  Counting the parameters is a def handed to `for_leaves` (`count_into`, into
  a one-entry array), where torch writes `sum(p.numel() for p in
  model.parameters())`: a closure cannot assign the running count it
  captured, and `leaves(model)` is not built.
- **run** (12 against 8): the configuration is one line,
  `GptConfig { vocab: len(vocab) }`, with the other sizes the record's field
  defaults; the seed is set before
  `Gpt(cfg)` draws, the parameter count takes two lines, and the losses are
  printed by a loop after training, where torch prints inside `train`.
- **loss** (3 against 0): `value_and_grad` differentiates a named def, so the
  loss is `sequence_loss`, where torch calls `F.cross_entropy` inside the
  training loop.
- **sampling** (10 against 7): the context is `take the last len(m.pos)
  elements of out`, where torch writes `out[-n:]` (a Viv slice does not clamp
  a negative start).
- **text as tokens** (4 against 4): `unique(TEXT)` for `sorted(set(TEXT))`,
  a map comprehension for `stoi`, and `encode` and `decode` bound to lambdas whose bodies are
  comprehensions, as the Python binds them.
- **checks** (12 against 4): each check is a named `test` block, where the
  Python writes one bare `assert`.
- **training loop** (18 against 21): `batch` cuts the windows with two
  comprehensions (the second stacks the windows into a batch-by-context
  matrix), as torch stacks a list of slices. `value_and_grad` returns the
  loss and the gradient as two names, so its step is braced, to store the
  loss.
- **block** (5 against 11), **attention** (9 against 14) and **forward** (8
  against 13) are shorter than torch's: the block is the paper's two lines,
  `X = X + self_attention(b.attn, rmsnorm(X))` and the MLP's, one step each,
  and lib/nn's `attention` returns only the head's output.
