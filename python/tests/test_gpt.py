# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vp.gpt against three references:

- the NumPy program study/compare/ml/tiny_gpt.py (hand-written backward
  pass and optimizers): its 30 losses, its trained matrices and its sample.
  The forward passes agree to the bit (the first six losses are equal);
  the gradients are summed in a different order (the compiler's backward
  pass against the hand-written one), so from step 7 the losses differ in
  the last bit or two, 1.2e-15 at most;
- a PyTorch reimplementation (python/tests/gpt_reference.py: autograd,
  scaled_dot_product_attention, torch.optim.Adam, float64): 50 losses at
  two sizes, to rounding;
- central differences of the loss: the Viv test of lib/gpt/gpt.viv, run
  through r.check().

    python -m pytest python/tests/test_gpt.py
"""
import contextlib
import dataclasses
import io
import re
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import vipyr as vp  # noqa: E402

TEXT = "hello world! " * 5
LONG = (
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of "
    "foolishness, it was the epoch of belief, it was the epoch of incredulity. "
)


@pytest.fixture(scope="module")
def twin():
    """study/compare/ml/tiny_gpt.py run as written: its globals."""
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(str(ROOT / "study" / "compare" / "ml" / "tiny_gpt.py"))


def test_numpy_twin(twin):
    r = vp.gpt.train(TEXT, steps=twin["STEPS"], seed=twin["INIT_SEED"])
    ref = np.array(twin["losses"])
    assert r.losses.shape == ref.shape
    assert np.array_equal(r.losses[:6], ref[:6])
    assert np.abs(r.losses - ref).max() <= 4 * np.finfo(float).eps * ref.max(), np.abs(r.losses - ref).max()
    assert r.model.size == twin["count"] == 6560
    g = vp.gpt.generate(r.model, twin["PROMPT"], n=twin["SAMPLES"], seed=twin["SAMPLE_SEED"])
    assert g.text == twin["written"]


def test_numpy_twin_weights(twin):
    """The trained matrices themselves, name by name, against the twin's."""
    r = vp.gpt.train(TEXT, steps=twin["STEPS"], seed=twin["INIT_SEED"])
    theirs = {_our_name(n): p for n, p, _ in twin["model"].named_parameters()}
    ours = dict(r.model.parameters())
    assert sorted(theirs) == sorted(ours)
    for name, w in ours.items():
        assert np.abs(w - theirs[name]).max() <= 1e-12 * np.abs(theirs[name]).max(), name


def _our_name(n):
    """tiny_gpt.py's parameter name as vp.gpt.Model's: blocks.0.attn.heads.1.q -> blocks[0].attn.heads[1].Wq."""
    n = re.sub(r"^embed\.", "", n)
    n = re.sub(r"\.(\d+)", r"[\1]", n)
    n = re.sub(r"\.out$", ".Wo", n)
    return re.sub(r"\.([qkv])$", lambda m: ".W" + m.group(1), n)


SIZES = [
    (TEXT, vp.gpt.GptConfig()),
    (
        LONG,
        vp.gpt.GptConfig(context=16, width=32, heads=4, layers=3, hidden=96, batch=8),
    ),
]


@pytest.mark.parametrize("text, cfg", SIZES)
def test_pytorch_reimplementation(text, cfg):
    torch = pytest.importorskip("torch")
    import gpt_reference as ref

    torch.set_num_threads(1)
    steps, seed = 50, 7
    r = vp.gpt.train(text, config=cfg, steps=steps, seed=seed)
    c = r.model.config
    ids = vp._gpt.tokenize(text)[1].astype(int)
    model = ref.TorchGpt(c, ref.init_arrays(c, seed))
    losses = ref.train(model, ids, steps)
    rel = np.abs(r.losses - losses) / np.abs(losses)
    assert rel.max() < 1e-12, rel.max()
    # the trained matrices too
    for name, w in r.model.parameters():
        t = model.p[name].detach().numpy()
        assert np.abs(w - t).max() <= 1e-11 * max(1.0, np.abs(t).max()), name


def test_check_runs_the_gradient_test():
    r = vp.gpt.train(TEXT, steps=3)
    rep = r.check()
    names = [t[0] for t in rep.tests]
    assert any("central differences" in n for n in names)
    assert rep.passed, rep


def test_loss_and_edits():
    r = vp.gpt.train(TEXT, steps=30, seed=7)
    m = r.model
    base = vp.gpt.loss(m, TEXT).loss
    import gpt_reference as ref

    torch = pytest.importorskip("torch")
    tm = ref.TorchGpt(m.config, dict(m.parameters()))
    ids = torch.tensor(vp._gpt.tokenize(TEXT)[1].astype(int))
    n = m.config.context
    xs = torch.stack([ids[s : s + n] for s in range(len(ids) - n)])
    ys = torch.stack([ids[s + 1 : s + 1 + n] for s in range(len(ids) - n)])
    assert abs(base - tm.loss(xs, ys).item()) <= 1e-12 * base
    m.blocks[0].mlp.up[:] = 0.0
    assert vp.gpt.loss(m, TEXT).loss != base


def test_untrained_loss_is_log_vocab():
    r = vp.gpt.train(TEXT, steps=1)
    assert abs(r.losses[0] - np.log(9.0)) <= 1e-15


def test_temperature_zero_is_greedy():
    m = vp.gpt.train(TEXT, steps=30, seed=7).model
    a = vp.gpt.generate(m, "hello", n=12, temperature=0.0, seed=1).text
    b = vp.gpt.generate(m, "hello", n=12, temperature=0.0, seed=2).text
    assert a == b and a.startswith("hello")


def test_token_ids():
    ids = np.array([0, 3, 1, 4, 2, 2, 0, 1, 3] * 4)
    r = vp.gpt.train(
        ids,
        config=vp.gpt.GptConfig(context=4, width=8, hidden=12, batch=2),
        steps=5,
    )
    assert r.model.vocab is None and r.model.config.vocab == 5
    g = vp.gpt.generate(r.model, ids[:3], n=4)
    assert g.text is None and len(g.tokens) == 7


def test_refusals():
    with pytest.raises(ValueError, match="multiple of heads"):
        vp.gpt.train(TEXT, config=vp.gpt.GptConfig(width=15, heads=2))
    with pytest.raises(ValueError, match="more than context"):
        vp.gpt.train("short", config=vp.gpt.GptConfig(context=8))
    m = vp.gpt.train(TEXT, steps=1).model
    with pytest.raises(ValueError, match="not in the model's vocabulary"):
        vp.gpt.generate(m, "xyz")
    with pytest.raises(ValueError):
        vp.gpt.Model.unpack(m.pack()[:-1], m.config)
    bad = dataclasses.replace(m)
    bad.tok = bad.tok[:, :3]
    with pytest.raises(Exception):
        vp.gpt.loss(bad, TEXT)


def test_fast_build_agrees_to_rounding():
    a = vp.gpt.train(TEXT, steps=30, seed=7)
    b = vp.gpt.train(TEXT, steps=30, seed=7, fast=True)
    assert np.abs(a.losses - b.losses).max() <= 1e-12
