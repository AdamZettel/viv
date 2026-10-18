# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vp.gguf on lib/gguf/tests/tiny.gguf: the metadata, the table, and every
tensor bit for bit against tiny.ref (gguf-py 0.19.0's dequantize, written by
tools/gguf/make_fixture.py); and against gguf-py itself when it is installed."""
from pathlib import Path

import numpy as np
import pytest

import vipyr as vp

ROOT = Path(__file__).resolve().parents[2]
TINY = ROOT / "lib/gguf/tests/tiny.gguf"
REF = ROOT / "lib/gguf/tests/tiny.ref"


def test_metadata_and_table():
    f = vp.gguf.read(TINY)
    m = f.metadata
    assert m["general.architecture"] == "tiny" and m["general.alignment"] == 64
    assert m["tiny.i8"] == -100 and m["tiny.u64"] == 2 ** 40 + 3 and m["tiny.bool"] is True
    assert m["tiny.str"] == "héllo ▁world 🙂"
    assert m["tiny.f32"] == np.float32(0.1) and m["tiny.f64"] == 2.0 ** -1074
    assert m["tiny.arr.str"] == ["a", "", "▁the", "<0x0A>"] and m["tiny.arr.nested"] == [1, 2, 3]
    assert f.alignment == 64 and f.data_offset == 1472 and len(f.tensors) == 8
    assert f.tensor_info("t.q4_k").shape == (8, 256) and f.tensor_info("t.q4_k").type == "Q4_K"


def test_every_tensor_bit_for_bit():
    f = vp.gguf.read(TINY)
    ref = np.fromfile(REF, dtype="<f4").reshape(len(f.tensors), -1)
    for k, t in enumerate(f.tensors):
        x = vp.gguf.tensor(TINY, t.name, file=f).array
        assert x.dtype == np.float32 and x.shape == t.shape
        assert np.array_equal(x.reshape(-1).view(np.uint32), ref[k].view(np.uint32)), t.name


def test_against_gguf_py():
    gguf = pytest.importorskip("gguf")
    r = gguf.GGUFReader(str(TINY))
    f = vp.gguf.read(TINY)
    for k, fld in r.fields.items():
        if not k.startswith("GGUF."):
            assert f.metadata[k] == fld.contents(), k
    for t in r.tensors:
        x = vp.gguf.tensor(TINY, t.name, file=f).array
        y = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32)), t.name


def test_checks_and_the_fixture_model():
    f = vp.gguf.read(TINY)
    assert f.check()
    r = vp.gguf.tensor(TINY, "t.q4_k", file=f)
    assert r.info.type == "Q4_K" and r.check()
    m = vp.gguf.read(ROOT / "lib/gguf/tests/stories260K.gguf")
    assert m.metadata["llama.embedding_length"] == 64 and len(m.tensors) == 48


def test_errors():
    with pytest.raises(KeyError):
        vp.gguf.tensor(TINY, "no such tensor")
