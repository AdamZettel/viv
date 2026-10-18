---
title: A small GPT in NumPy and in Viv
order: 1
output: identical
---
- The Viv column is the program `lib/gpt/story_gpt.viv` runs, with its
  `#! use` lines expanded by hand: `lib/gpt/gpt.viv` and the defs it calls from
  `lib/nn`, `lib/polar`, `lib/optim` and `lib/random`, written in the same
  style as `lib/gpt` (one-line steps, comprehensions for
  arrays of records and for the batch). Defs are moved so that each layer
  sits in the same region as in the Python classes. Three edits: every step's
  loss is printed to six decimals; the optimizers' settings, records in
  `lib/optim`, are constants here; and Newton-Schulz reads Muon's
  coefficients and step count from those constants, as the Python's reads
  them from its defaults, where `lib/polar`'s takes them as arguments. The NumPy
  program writes its layers and its optimizer out, so this program does too;
  the PyTorch pair ([tiny_gpt_torch](/compare/ml/tiny_gpt_torch)) imports them
  from `lib/nn`, `lib/optim` and `lib/random`, as the PyTorch program imports
  torch.
- The Python column is what a NumPy programmer would write from scratch: a
  `Config` dataclass, one class per layer (`Embedding`, `Head`, `Attention`,
  `MLP`, `Block`, `GPT`), each holding its weights in a dict and caching what
  its `backward` needs on `self`, and an optimizer that walks
  `named_parameters()`. The Viv model is the same tree as records: `Gpt { tok,
  pos, blocks: [Block], unembed }`, `Block { attn, mlp }`, `Attention { heads:
  [Head], Wo }`, `Head { Wq, Wk, Wv }`, `Mlp { up, down }`, and a matrix is read
  where it is (`b.mlp.up`, `h.Wq`). The Python backward pass is written by
  hand; the Viv program's is `value_and_grad(sequence_loss, model, ...)`, which
  the compiler writes from the forward pass and which gives a `Gpt` of
  gradients. The Viv optimizer updates the model in place, `for_leaves(muon_update,
  model.blocks, g.blocks, m.blocks, lr_scale)` for the blocks' matrices and
  `adam_update` for the other three, where the Python walks
  `named_parameters()`. Both train the same model (6,560 parameters, 2 blocks,
  2 heads, width 16) for 30 steps with Muon on the matrices inside the blocks
  and Adam on the embeddings and the output map, then sample 24 characters.
- The two programs print the same text. Printed with 17 digits, the 30 losses
  differ by at most 8.9e-16, and the sampled text is the same string. The
  weights are the same because the Python side replays Viv's generator
  (`Minstd`: Park-Miller MINSTD, Box-Muller for the normals, the inverted
  cumulative sum for the sample), not `np.random.default_rng`: `lib/gpt` draws
  with `srand` and `rand`, not with `lib/nprandom`. Both sum a row along its
  last axis in NumPy's pairwise order (Viv's `sum`, `mean` and `amax` along an
  axis follow NumPy's). The rest of the difference is the order of the sums in
  `@` (NumPy calls BLAS, which blocks its sums its own way), the order of the
  hand-written backward pass against the generated one, and the last bit of
  `exp`, `log` and `cos`, which Viv computes with its own libm.
- A third program, `tiny_gpt.phrased.viv` (the "Viv in phrases" column), is
  the Viv column with the phrases `lib/gpt`, `lib/nn` and `lib/random` declare
  and use: `take the causal mask of size T`,
  `take the softmax of each row of S`, `take the cross-entropy of
  gpt_forward(m, x) against y`, `take the last len(m.pos) elements of out`.
  Each phrase is the call it stands for, so it prints the same bytes. The phrases that looked a matrix up in a
  flat parameter vector (`take the query matrix of this head`) are gone with
  the vector: the code reads the matrix where it is.
- Both programs batch the four sequences of a step into one `(4, 8, 16)`
  array and let `@` and the reductions work along the last axes: `@` on a
  stack of matrices multiplies each matrix of the stack, as NumPy's does, so
  `train` calls `value_and_grad` once per step, on the whole batch.

### Measured

Lines are code lines: blank lines, comments and docstrings are not counted;
characters are those lines' characters. A lookup is one place the program
finds a weight or a size: in Python `params["q"]` or `cfg.width`; in Viv a
weight read from its record by name (`h.Wq`, `b.mlp.up`, `m.tok`), or a read
of a size from the configuration record (`c.width`). `study/compare/measure.py`
counts them, and `check.sh` fails when these tables are out of date.

%% measured
| region | Python lines | Viv lines | Viv / Python | Python lookups | Viv lookups | Viv in phrases lines | Viv in phrases lookups |
|---|---|---|---|---|---|---|---|
| imports | 3 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| config | 23 | 15 | 0.7 | 0 | 0 | 15 | 0 |
| random draws | 15 | 19 | 1.3 | 0 | 0 | 19 | 0 |
| parameters and init | 14 | 25 | 1.8 | 0 | 19 | 26 | 19 |
| embedding | 14 | 3 | 0.2 | 10 | 2 | 3 | 2 |
| attention | 41 | 29 | 0.7 | 15 | 4 | 28 | 4 |
| block | 35 | 16 | 0.5 | 10 | 2 | 16 | 2 |
| forward | 15 | 8 | 0.5 | 5 | 1 | 8 | 1 |
| loss | 11 | 9 | 0.8 | 0 | 0 | 9 | 0 |
| backward | 6 | 0 | 0.0 | 1 | 0 | 0 | 0 |
| optimizer | 27 | 38 | 1.4 | 0 | 12 | 39 | 12 |
| training loop | 15 | 18 | 1.2 | 4 | 4 | 21 | 4 |
| sampling | 6 | 10 | 1.7 | 1 | 1 | 10 | 1 |
| text as tokens | 6 | 4 | 0.7 | 0 | 0 | 7 | 0 |
| run | 7 | 12 | 1.7 | 0 | 0 | 21 | 0 |
| checks | 4 | 12 | 3.0 | 0 | 0 | 12 | 0 |
| **all** | **242** | **218** | **0.9** | **46** | **45** | **234** | **45** |

| characters of code | Python | Viv | Viv / Python | Viv in phrases |
|---|---|---|---|---|
| imports | 62 | 0 | 0.0 | 0 |
| config | 423 | 338 | 0.8 | 312 |
| random draws | 497 | 291 | 0.6 | 291 |
| parameters and init | 466 | 937 | 2.0 | 937 |
| embedding | 524 | 99 | 0.2 | 99 |
| attention | 1518 | 706 | 0.5 | 854 |
| block | 1224 | 339 | 0.3 | 339 |
| forward | 493 | 175 | 0.4 | 175 |
| loss | 391 | 359 | 0.9 | 377 |
| backward | 240 | 0 | 0.0 | 0 |
| optimizer | 908 | 1261 | 1.4 | 1346 |
| training loop | 596 | 627 | 1.1 | 656 |
| sampling | 197 | 290 | 1.5 | 287 |
| text as tokens | 167 | 159 | 1.0 | 193 |
| run | 322 | 398 | 1.2 | 503 |
| checks | 176 | 469 | 2.7 | 469 |
| **all** | **8204** | **6448** | **0.8** | **6838** |

| the forward and backward defs | Python | Viv |
|---|---|---|
| defs | 16 | 11 |
| parameters, all of them together | 19 | 21 |
| output parameters (arrays the def itself writes results into) | 0 | 0 |
| most in one def | 3 (`rmsnorm_backward`) | 4 (`attention`) |
%% end

### Where the extra Viv lines come from

- **The heads.** Each head has its own record of three matrices, so
  `self_attention` loops over the heads and stores each one's output in its
  columns of `Z` (`Z[:, :, i * hd:(i + 1) * hd] = attention(N, h.Wq, h.Wk,
  h.Wv)`), where the Python concatenates a list. The rest of the forward pass
  works on the whole `(4, 8, 16)` batch at once, as the Python's does: `@`
  multiplies each matrix of a stack, RMSNorm, the softmax and the
  cross-entropy reduce along the last axis with keepdims, and `value_and_grad`
  has the adjoints of those reductions and of the broadcasts between arrays
  of different ranks.
- **The configuration.** The sizes are `GptConfig`'s field defaults, so the
  run region writes `GptConfig { vocab: len(vocab) }`, as the Python writes
  `Config(vocab=len(vocab))`.
- **No default arguments.** `def f(x, y = 2.0)` does not parse, so the
  optimizers' constants are top-level names that `newton_schulz`,
  `muon_update` and `adam_update` read.
- **The optimizer by hand.** Muon and Adam are written out per matrix
  (`muon_update`, `adam_update`) and applied with `for_leaves`; the Python
  walks `named_parameters()` with the same arithmetic. The optimizer region
  also holds Newton-Schulz, as the Python's does.
- **What costs the same or less.** The model is a tree of records, as the
  Python's is a tree of modules; the configuration is a record (`GptConfig`);
  the backward pass is `value_and_grad` (the backward region has no Viv code,
  against Python's 6 lines of hand-written backward calls); the heads'
  outputs go into their columns with one slice store. The embedding is one
  `take` along the rows of the table. The embedding, attention, the block and
  the forward pass take fewer characters than in NumPy.
