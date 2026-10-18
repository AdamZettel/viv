# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
A small GPT trained on one paragraph
====================================

`vp.gpt.train` trains the GPT of `lib/gpt/gpt.viv` (nanoGPT's layout, the
modded-nanogpt speedrun's Muon and Adam) on a paragraph of Dickens, one
character per token. The training loop is compiled Viv; the model comes
back as named NumPy arrays, which `generate`, `attention` and `loss` read
again on every call. The same run in PyTorch (float64, autograd,
`torch.optim.Adam`) gives the same losses to 14 digits
(`python/tests/test_gpt.py`).
"""

# %%
# ## Training
#
# Three blocks of four heads, 32 numbers per position, a context of 16
# characters, batches of 8 windows, 1000 steps. The first loss is log 27
# exactly: the output map starts at zero, so the first prediction is
# uniform over the 27 characters of the text.

import numpy as np
import vipyr as vp
from vipyr.figure import figure, grid

text = (
    "It was the best of times, it was the worst of times, it was the age "
    "of wisdom, it was the age of foolishness, it was the epoch of belief, "
    "it was the epoch of incredulity, it was the season of Light, it was "
    "the season of Darkness. "
)
cfg = vp.gpt.GptConfig(
    context=16, width=32, heads=4, layers=3, hidden=96, batch=8
)
r = vp.gpt.train(text, config=cfg, steps=1000, seed=7)
m = r.model
print(
    "%d parameters, %d characters" % (m.size, m.config.vocab),
    "| first loss %.6f (log %d = %.6f)"
    % (r.losses[0], m.config.vocab, np.log(m.config.vocab)),
    "| last loss %.4f" % r.losses[-1],
)
print(r.check().passed)

fig = figure("Training loss (one batch per step)", "step", "loss (nats)")
fig.yscale("log")
fig.plot(np.arange(1, len(r.losses) + 1), r.losses, "batch loss")
fig.style("batch loss", "s1 nomarker")

# %%
# ## The model as named arrays
#
# Every matrix is where the Viv record keeps it. A block's attention has
# one query, key and value matrix per head and one output map for all
# heads together.

for name, w in list(m.parameters())[:8]:
    print("%-28s %s" % (name, w.shape))
print("...")
print("%-28s %s" % ("unembed", m.unembed.shape))

# %%
# ## Writing
#
# From the prompt "it was the", at three temperatures. At 0 the model takes
# its most likely character every time. A model this size trained on one
# paragraph has learned its words and phrases, not yet the paragraph.

for t in (0.0, 0.7, 1.0):
    g = vp.gpt.generate(m, "it was the", n=80, temperature=t, seed=11)
    print("T = %.1f: %r" % (t, g.text))

# %%
# ## Where the heads look
#
# The attention weights of the last block for the prompt "the age of
# wisd": row i is how much position i takes from each earlier position.

a = vp.gpt.attention(m, "the age of wisd")
labels = a.labels
g = grid("Attention of block 3 for 'the age of wisd'", 2, 2)
for h in range(cfg.heads):
    f = g.figure("head %d" % (h + 1), "from position", "at position")
    f.heatmap(a.weights[-1, h], "weight")
    f.clim(0.0, 1.0)
    for i, ch in enumerate(labels):
        f.xtick(repr(ch)[1:-1], i)
        f.ytick(repr(ch)[1:-1], i)

# %%
# ## Taking a head out
#
# The output map of block l takes the heads' results side by side, so the
# rows `h * d` to `(h + 1) * d` of `blocks[l].attn.Wo` carry head h. Zeroing
# them removes the head; `loss` reads the edited arrays. The loss is the
# mean over every window of the text.

base = vp.gpt.loss(m, text).loss
d = cfg.width // cfg.heads
print("all heads: loss %.4f" % base)
for layer in range(cfg.layers):
    rise = []
    for h in range(cfg.heads):
        Wo = m.blocks[layer].attn.Wo
        kept = Wo[h * d : (h + 1) * d].copy()
        Wo[h * d : (h + 1) * d] = 0.0
        rise.append(vp.gpt.loss(m, text).loss - base)
        Wo[h * d : (h + 1) * d] = kept
    print(
        "block %d, loss added by removing each head: %s"
        % (layer + 1, "  ".join("%.3f" % x for x in rise))
    )
