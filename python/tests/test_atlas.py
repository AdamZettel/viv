# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vipyr.atlas (`viv atlas`, lang/viv/src/atlas.rs) on the MINPACK story
program: the text, the JSON with the run's call counts, and the offline page."""
import re

import vipyr as vp

PROGRAM = "lib/minpack/story_rosenbrock.viv"


def test_text_has_the_modules_and_the_call_tree_with_counts():
    t = vp.atlas(PROGRAM, text=True)
    assert t.startswith("atlas of lib/minpack/story_rosenbrock: ")
    assert " in 8 modules" in t
    assert "\n      lmder  x1\n" in t
    assert "not run" in vp.atlas(PROGRAM, text=True, run=False)


def test_json_carries_the_calls_of_the_run_and_the_map():
    d = vp.atlas(PROGRAM, json=True)
    calls = {(a, b): n for a, b, n in d["calls"]}
    assert calls[("lmder1", "lmder")] == 1
    assert calls[("lmder", "lmpar")] > 0
    assert len(d["map"]["modules"]) == 8
    assert vp.atlas(PROGRAM, json=True, run=False)["calls"] is None


def test_html_is_one_file_that_loads_nothing(tmp_path):
    out = vp.atlas(PROGRAM, out=tmp_path / "rosen.html")
    html = open(out, encoding="utf-8").read()
    assert 'id="atlas-data"' in html
    assert "<script src" not in html and "<link" not in html
    assert all(u.startswith("http://www.w3.org/") for u in re.findall(r"https?://[^\s\"'<>)]+", html))


def test_a_command_past_its_timeout_is_an_error_not_a_hang():
    import pytest

    with pytest.raises(RuntimeError, match="still running after"):
        vp.atlas(PROGRAM, text=True, timeout=0.001)
