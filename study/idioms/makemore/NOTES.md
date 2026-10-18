# makemore MLP in Viv

## Source

- https://github.com/karpathy/makemore @ `988aa59e4d8fefa526d06f3b453ad116258398d4`
  (shallow clone at `viv-wt/src/makemore`, not copied here).
- Ported: `create_datasets`, `CharDataset`, `MLP` (Bengio et al. 2003), the AdamW
  training loop, `evaluate`, `generate` and `print_samples`, for `--type mlp`.

## Files

| file | what |
|---|---|
| `makemore.viv` | the port: file text to words, vocabulary, split, encoding, MLP forward and backward, AdamW, evaluation, sampling, and 5 tests against the reference |
| `names_subset.viv` | generated: the first 300 lines of `names.txt` as `chars("...")` (Viv cannot read a file) |
| `gen_names.py` | writes `names_subset.viv` |
| `reference.py` | imports makemore's own `MLP`, `CharDataset`, `ModelConfig` from the clone and trains with `torch.optim.AdamW`, float64 |
| `reference_output.txt`, `viv_output.txt` | the two runs |

Run: `viv run ./makemore.viv` (the `./` matters, see F19), `python3 reference.py`.

## Sizes

First 300 names (max length 10, 26 letters, 270 train / 30 test), `block_size` 11
(makemore's MLP uses `max_word_length + 1` as its context), `n_embd` 8,
`n_embd2` 32, batch 16, 300 steps, AdamW lr 1e-2, weight decay 0.01, betas
(0.9, 0.99), eps 1e-8. 3963 parameters. Algorithm unchanged from makemore.

## Shared randomness

Both programs draw from Viv's Park-Miller generator (`srand(3407)`,
reimplemented in `reference.py`), so they see the same split, weights, batches
and samples:

| makemore | here (both programs) |
|---|---|
| `torch.randperm` | Fisher-Yates |
| torch's init: `Embedding` N(0,1), `Linear` U(-1/sqrt(fan_in), +) | the same distributions; N(0,1) by Box-Muller |
| `RandomSampler(replacement=True)` | `BATCH` uniform draws |
| `torch.multinomial` | inverse CDF on the softmax |
| `evaluate`: shuffled loader, batch 100, 10 batches | whole split in order, batches of 50, mean of batch means |

The Viv port runs only the positions whose target is not -1 through the
network. makemore runs all of them and `cross_entropy(ignore_index=-1)` drops
them; the loss and gradient are identical.

## Agreement

All within **1e-12** (the tests in `makemore.viv` assert it against 15-digit
values from `reference.py`):

| quantity | torch (makemore's MLP) | Viv |
|---|---|---|
| train / test loss before training | 3.314562857124737 / 3.297041321791764 | pass at 1e-12 |
| batch loss at steps 0, 50, 100, 150, 200, 250, 299 | 3.3506278 … 1.6317912 | pass at 1e-12 |
| train / test loss after 300 steps | 1.647162050005586 / 2.529329933150971 | pass at 1e-12 |
| 10 samples | arena, dra, lucia (train), elaale, rysorno, jutbetoja, joba, matrerani, zhanante, briella (train) | identical |

## Both engines

`VIV=viv-wt/bin/viv tools/both-engines.sh study/idioms/makemore makemore`:
`5 tests pass, 0 failed; identical output`. Interpreter 12.9 s CPU, native C
(`cc -O2`) 0.16 s, torch float64 about 2.8 s CPU including import. The direct
wasm backend compiles it (88 KB module); it was not run, because
`tools/three-engines.sh` needs a `viv.wasm` built with cargo, which this job may
not run.

Before two workarounds the program did not run the same in both engines: see F5
(short-circuit) and F6 (the name collision that made the C not build).

## Line counts (ported part only; code lines, no blanks, comments or docstrings)

- makemore.py, the ported functions and loop: 145
- makemore.viv: 389, plus 30 lines of tests. About 110 of the difference is the
  backward pass and the sampling/decoding helpers that torch and Python strings
  provide; about 100 is F1 to F4 and F9 (text handling done on arrays of codes).

## Friction ledger

Severity: 1 cosmetic, 2 costs lines, 3 blocks or distorts the algorithm.

### F1. No file input (strings/IO, sev 3, once)

A program cannot open `names.txt`. The file is pasted into a generated module
as a `chars("...")` literal by `gen_names.py`, and a different dataset or subset
means regenerating source. `lib/corpus.viv` does the same.

- Had to: `#! use names_subset` then `text = names_file()`
- Wanted: `text = read_text("names.txt")` or `lines = read_lines("names.txt")`
- Python: `data = open(input_file).read()`

### F2. No string type, no list of strings (missing data structure, sev 2, 6 sites)

Words are rows of a zero-padded code matrix plus a `word_len` vector. Splitting
and stripping is a 35-line `line_spans` that records start and end offsets.
Every string operation is a hand loop: `word(w)` to cut off padding, `contains`
(15 lines), `decode`, `concat`.

- Had to (the ugliest part):
  ```
  def line_spans(text: Arr, spans: Arr) {
      ...
          if c == NEWLINE {
              lo = start
              hi = i
              while lo < hi {
                  if not is_space(text[lo]) {
                      break
                  }
                  lo = lo + 1
              }
              ...   # the same from the right
              if hi > lo {
                  spans[n, 0] = lo
                  spans[n, 1] = hi
                  n = n + 1
  ```
- Wanted: `words = [strip(w) for w in split(text, "\n") if w != ""]`, and `name in train_words`
- Python: `words = [w.strip() for w in data.splitlines()]; words = [w for w in words if w]`

### F3. No growable array / append (missing data structure, sev 2, 4 sites)

Anything whose size is found while scanning takes two passes (count, then
allocate and fill) or an oversized buffer: `spans = zeros(len(text), 2)`;
`distinct` counts the unique values, then loops again to write them;
`count_positions` has to run before `contexts` to size the target vector.

- Wanted: `out = []` then `push(out, x)`
- Python: `list.append`

### F4. No set or dict (missing data structure, sev 2, 2 sites)

`sorted(set(...))` is a flatten, `sort`, and two dedupe loops (25 lines).
`stoi` is a table indexed by character code, `zeros(max(vocab) + 1)`, which
works for ASCII and would be a million entries for a CJK name.

- Wanted: `vocab = unique(words)`; `stoi = dict(...)`
- Python: `chars = sorted(list(set(''.join(words))))`; `{ch: i + 1 for i, ch in enumerate(chars)}`

### F5. `and` / `or` do not short-circuit in the interpreter; they do in C (engine gap, sev 3, 3 sites)

`if i == len(text) or text[i] == NEWLINE` stops the interpreter with "index 2085
out of bounds"; the same program compiled to C runs. Minimal case:

```
a = [1.0]
i = 1
if i == 1 or a[i] == 0 {    # interpreter: error: index 1 out of bounds; native: prints 1
```

The guard idiom `lo < hi and is_space(text[lo])` and
`i == 0 or flat[i] != flat[i - 1]` fail the same way. The two engines disagree
on a program with no error in it.

- Had to: read `text[i]` into `c` under an `if`; turn each guarded `while` into `while lo < hi { if not ... { break } ... }`; rewrite dedupe with a `prev` variable
- Wanted: `while lo < hi and is_space(text[lo])`
- Python / Julia: `and`/`&&` short-circuit

### F6. One global namespace across `#! use`; the collision passes the interpreter and breaks the C (engine gap / tooling, sev 2, once)

`#! use lib/optim` splices `lib/polar`, `lib/svd` and `lib/schur`, and
`lib/schur` has `def block_size(T, k)`. My global `block_size` (makemore's own
name for the context length) ran fine in the interpreter; the C compile then
failed with `error: redefinition of 'block_size' as different kind of symbol`.
Renamed to `BLOCK`.

- Wanted: module namespaces (`optim.adam`), or at least a Viv-level error in both engines naming the two definitions
- Python: `from torch.optim import AdamW` imports one name

### F7. No broadcasting of a row vector over a matrix (missing builtin, sev 2, 4 sites)

`A + b` with `A` n by m and `b` of length m (or 1 by m) is "array shape
mismatch". A bias is added as an outer product with a column of ones, and the
bias gradient (a column sum; there is no `sum(a, axis)`) is the same trick
transposed.

- Had to: `tanh(X @ W1 + ones_column(n) @ b1)`; `param_add(grad, model, B1, transpose(ones) @ dpre)`
- Wanted: `tanh(X @ W1 + b1)`; `sum(dpre, 0)`
- Python: `x @ W + b` (numpy/torch broadcasting); `dpre.sum(0)`

### F8. No autodiff for array code (missing builtin, sev 2, once)

The backward pass is written by hand, 25 lines: cross-entropy (from `lib/nn`),
two matmul adjoints, the tanh derivative `(1 - H * H)`, and the embedding
scatter. It is exact (the losses agree with torch to 1e-12 through 300 steps),
but each new model means a new backward.

- Wanted: `grad = gradient(loss_of, theta)`
- Python: `loss.backward()`

### F9. No slicing (missing builtin, sev 2, 6 sites)

`train_ids = perm[:n_train]`, a batch of the split, a name without its padding,
and joining two arrays all need a `zeros` plus a loop: the split (2 loops), the
evaluation batch, `word(w)`, `concat`, `decode`.

- Wanted: `perm[0:n_train]`, `concat(a, b)`
- Python: `rp[:-test_set_size]`, `a + b`

### F10. One return value (type system, sev 2, 2 sites)

`contexts` has to hand back both the context matrix and the targets; the
targets come back through an out-parameter whose size the caller must compute
first with `count_positions`. `lib/nn` says the same about backward passes.

- Wanted: `ctx, target = contexts(ids)`
- Python: `return x, y`

### F11. No scatter-add (missing builtin, sev 2, once)

`take` gathers the embedding rows in one call, but its inverse `put` overwrites,
so repeated tokens lose gradient. The embedding gradient is a triple loop.

- Wanted: `put_add(grad, flat, dX)`
- Python: `index_add_`, or autograd through `nn.Embedding`

### F12. No `randn`, random integer, shuffle or categorical draw (missing builtin, sev 2, 3 sites)

Box-Muller (`randn`), Fisher-Yates, and the inverse-CDF `draw` are written out.
Each is short, but every ML port will write them again.

- Wanted: `randn()`, `shuffle(a)`, `choice(p)`
- Python: `torch.randn`, `torch.randperm`, `torch.multinomial`

### F13. `*=` on an array rebinds instead of writing in place (type system, sev 2, once)

`a *= 3.0` makes a new array: an alias `b = a` keeps the old values, and inside
a def the caller's array is untouched. NumPy and Julia (`.*=`) write in place.
At top level it happened to do what AdamW's decay needs:
`theta *= 1.0 - LR * WD`.

- Wanted: in-place `a .*= s` or documented rebinding
- Python: `p.mul_(1 - lr * wd)`

### F14. `lib/optim`'s Adam has no weight decay (missing builtin, sev 1, once)

AdamW is the decay line above plus `optim_step`. The rest of Adam matched torch
to 1e-12.

### F15. Output: 6 significant digits, no formatting, `text:` prefix on every line (strings/IO, sev 2, 12 sites)

`show` prints `%g`. The only way to confirm agreement past six digits is a
`test` with the reference's numbers written into the program. A line cannot mix
text and a number: `print(f"step {step} | loss {loss:.4f}")` is not
expressible, and sampled names print as `text: arena (new)` after a
hand-written `concat(name, chars(" (new)"))`.

- Wanted: `print("step ", step, " loss ", fmt(loss, 12))`, `print(name, " (new)")`
- Python: f-strings

### F16. `for` counts up only (control flow, sev 1, once)

Fisher-Yates walks down: `for k to n_words - 1 { i = n_words - 1 - k ... }`.

- Wanted: `for i from n - 1 down to 1`
- Python: `for i in range(n - 1, 0, -1)`

### F17. Records do not run in the interpreter (engine gap, sev 1, once)

The model could not be one value with named tensors; `lib/params` (a flat
vector plus a layout table) stands in, and `param_get` copies each weight
matrix out on every forward.

### F18. Undocumented builtins (tooling, sev 1, 3)

`text`, `chars` and `shape` are not in the reference page's builtin tables.
`text` was found by reading `interp.rs`.

### F19. `viv run makemore.viv` from its own directory cannot find `lib/` (tooling, sev 1, every run)

"no module by that name (looked for lib/nn.viv beside the program and under )":
the root search starts from an empty parent path. `viv run ./makemore.viv` works.

### F20. Error locations are in the spliced source (tooling/error message, sev 2, every error)

The out-of-bounds error in F5 was reported at `./makemore.viv:1983` in a
592-line file: the line number counts the text of every `#! use`d module
spliced in front.

## Wins

- `lib/params` + `lib/optim`: five tensors in one vector, and Adam plus one decay
  line reproduced `torch.optim.AdamW` to 1e-12 over 300 steps.
- `lib/nn`'s `cross_entropy_fwd` / `cross_entropy_bwd` and `softmax_fwd`
  dropped in unchanged.
- The forward pass is as short as torch's: `tanh(X @ W1 + ...)`, `H @ W2 + ...`,
  elementwise `tanh`, `*`, `1.0 - H * H`, `transpose`.
- `take(theta, flat)` gathers and concatenates all embeddings in one call.
- Both engines byte-identical once F5 and F6 were worked around; native C is
  about 80 times faster than the interpreter.
- `test` blocks put the reference's numbers next to the code that has to match
  them, and the process exits nonzero if one drifts.
- `chars("...")` makes a code array from a literal, and `text()` prints one.
- Globals read inside defs (`word_len`, `xs`, `model`) the way module-level
  names work in Python.
- `-> Bool` functions with `not` read cleanly (`is_space`, `contains`).
