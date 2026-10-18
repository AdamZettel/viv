# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The reference run for makemore.viv: karpathy/makemore's own MLP, CharDataset
and ModelConfig (imported from the clone, not copied), trained with torch's
AdamW at the same tiny size as the Viv port.

Every random draw comes from Viv's generator (Park-Miller, 16807 mod 2^31-1,
seeded by srand), reimplemented here, so both programs see the same split,
the same initial weights, the same batches and the same samples:

  - the train/test split is a Fisher-Yates shuffle (makemore: torch.randperm)
  - weights: the embedding is N(0,1) by Box-Muller, each Linear U(-1/sqrt(fan_in), +)
    (the same distributions torch uses; makemore leaves torch's init alone)
  - a batch is BATCH draws with replacement (makemore: RandomSampler)
  - sampling is inverse-CDF on softmax probs (makemore: torch.multinomial)
  - evaluation is the mean of per-batch means over the whole split, in order,
    batches of 50 (makemore: shuffled DataLoader, batch 100, max 10 batches)

float64 throughout, so the numbers should agree with Viv to rounding.

    python3 reference.py
"""
import math
import os
import sys
import types

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CLONE = os.path.join(HERE, "..", "..", "..", "..", "src", "makemore")
sys.modules.setdefault("torch.utils.tensorboard", types.SimpleNamespace(SummaryWriter=None))
sys.path.insert(0, CLONE)
from makemore import MLP, CharDataset, ModelConfig  # noqa: E402

torch.set_default_dtype(torch.float64)

# ---- the sizes (identical constants at the top of makemore.viv) ----
NW = 300          # first NW lines of names.txt
N_EMBD = 8
N_EMBD2 = 32
BATCH = 16
STEPS = 300
LR = 1.0e-2
WD = 0.01
SEED = 3407
EVAL_BATCH = 50
NUM_SAMPLES = 10


class ParkMiller:
    """Viv's srand/rand, as the C runtime writes them."""

    def __init__(self):
        self.state = 12345.0

    def srand(self, s):
        m = 2147483646.0
        v = abs(s)
        v = v - m * math.trunc(v / m)
        self.state = math.trunc(v) + 1.0

    def rand(self):
        p = 16807.0 * self.state
        self.state = p - 2147483647.0 * math.trunc(p / 2147483647.0)
        return self.state / 2147483647.0


rng = ParkMiller()
rng.srand(SEED)

# ---- the data, as makemore's create_datasets does it ----
with open(os.path.join(CLONE, "names.txt")) as f:
    data = f.read()
words = data.splitlines()[:NW]
words = [w.strip() for w in words]
words = [w for w in words if w]
chars = sorted(list(set("".join(words))))
max_word_length = max(len(w) for w in words)
print(f"number of examples in the dataset: {len(words)}")
print(f"max word length: {max_word_length}")
print(f"number of unique characters in the vocabulary: {len(chars)}")
print("vocabulary:")
print("".join(chars))

test_set_size = min(1000, int(len(words) * 0.1))
rp = list(range(len(words)))
for i in range(len(words) - 1, 0, -1):
    j = int(rng.rand() * (i + 1))
    rp[i], rp[j] = rp[j], rp[i]
train_words = [words[i] for i in rp[:-test_set_size]]
test_words = [words[i] for i in rp[-test_set_size:]]
print(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")
print(f"first test word: {test_words[0]}")
train_dataset = CharDataset(train_words, chars, max_word_length)
test_dataset = CharDataset(test_words, chars, max_word_length)

vocab_size = train_dataset.get_vocab_size()
block_size = train_dataset.get_output_length()
config = ModelConfig(vocab_size=vocab_size, block_size=block_size, n_embd=N_EMBD, n_embd2=N_EMBD2)
model = MLP(config)

# ---- the initial weights, drawn in the order makemore.viv draws them ----
def normal():
    u1 = rng.rand()
    u2 = rng.rand()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def uniform_matrix(rows, cols, fan_in):
    bound = 1.0 / math.sqrt(fan_in)
    return torch.tensor([[(2.0 * rng.rand() - 1.0) * bound for _ in range(cols)] for _ in range(rows)])


with torch.no_grad():
    wte = torch.tensor([[normal() for _ in range(N_EMBD)] for _ in range(vocab_size + 1)])
    model.wte.weight.copy_(wte)
    fan1 = block_size * N_EMBD
    model.mlp[0].weight.copy_(uniform_matrix(fan1, N_EMBD2, fan1).T)   # Viv stores W as (in, out)
    model.mlp[0].bias.copy_(uniform_matrix(1, N_EMBD2, fan1)[0])
    model.mlp[2].weight.copy_(uniform_matrix(N_EMBD2, vocab_size, N_EMBD2).T)
    model.mlp[2].bias.copy_(uniform_matrix(1, vocab_size, N_EMBD2)[0])
print(f"model #params: {sum(p.numel() for p in model.parameters())}")


def stack(dataset, idx):
    xs, ys = zip(*[dataset[i] for i in idx])
    return torch.stack(xs), torch.stack(ys)


@torch.inference_mode()
def evaluate(dataset):
    losses = []
    for start in range(0, len(dataset), EVAL_BATCH):
        X, Y = stack(dataset, range(start, min(start + EVAL_BATCH, len(dataset))))
        _, loss = model(X, Y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


print(f"step 0 train loss {evaluate(train_dataset):.15f} test loss {evaluate(test_dataset):.15f}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.99), eps=1e-8)
for step in range(STEPS):
    idx = [int(rng.rand() * len(train_dataset)) for _ in range(BATCH)]
    X, Y = stack(train_dataset, idx)
    logits, loss = model(X, Y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 50 == 0 or step == STEPS - 1:
        print(f"step {step} | loss {loss.item():.15f}")

print(f"step {STEPS} train loss {evaluate(train_dataset):.15f} test loss {evaluate(test_dataset):.15f}")


# ---- sampling: makemore's generate(), with multinomial as an inverse CDF ----
@torch.no_grad()
def generate_one(steps):
    idx = torch.zeros(1, 1, dtype=torch.long)
    for _ in range(steps):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        probs = F.softmax(logits[:, -1, :], dim=-1)[0]
        u = rng.rand()
        cum = 0.0
        nxt = vocab_size - 1
        for k in range(vocab_size):
            cum += probs[k].item()
            if u < cum:
                nxt = k
                break
        idx = torch.cat((idx, torch.tensor([[nxt]])), dim=1)
    return idx[0, 1:].tolist()


print("samples:")
for _ in range(NUM_SAMPLES):
    row = generate_one(train_dataset.get_output_length() - 1)
    crop = row.index(0) if 0 in row else len(row)
    word = train_dataset.decode(row[:crop])
    tag = "in train" if train_dataset.contains(word) else ("in test" if test_dataset.contains(word) else "new")
    print(f"  {word} ({tag})")
