# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The atlas of a Viv program from Python: its defs by module, the calls
between them and how many times each ran in one run.

    vp.atlas("lib/minpack/story_rosenbrock.viv")             # writes story_rosenbrock.atlas.html
    print(vp.atlas("lib/minpack/story_rosenbrock.viv", text=True))
    data = vp.atlas("lib/minpack/story_rosenbrock.viv", json=True)
    data["calls"]                                             # [[caller, callee, count], ...]

`vp.atlas` runs `viv atlas` (lang/viv/src/atlas.rs). The HTML file holds the
data and its viewer and opens with no network.
"""
from __future__ import annotations

import json as _json
import os
import subprocess

from . import _core, _spec

__all__ = ["atlas"]


# Seconds `vp.atlas` waits for `viv atlas`. The command's own run stops past
# its step budget (main.rs ATLAS_STEP_BUDGET); this catches what is left, such
# as a program whose build hangs.
TIMEOUT_SECONDS = 120.0


def atlas(path, out=None, text=False, *, json=False, run=True, timeout=TIMEOUT_SECONDS):
    """The atlas of the .viv file at `path` (a repository-relative path works
    in a checkout). With `text`, the atlas as text (a str); with `json`, the
    explorer bundle with the run's call counts and the code map (a dict);
    otherwise the HTML file is written to `out` (default
    `<name>.atlas.html` in the working directory) and its path returned.
    `run=False` skips the run: the calls are the ones the text makes. A run
    past its step budget stops and the atlas shows the calls the text makes;
    `timeout` (seconds, None for no limit) bounds the whole command."""
    if text and json:
        raise ValueError("vp.atlas: text=True or json=True, not both")
    p = str(_spec.source(path)) if not os.path.isabs(str(path)) and not os.path.exists(str(path)) else str(path)
    args = [_core.compiler(), "atlas", p]
    if not run:
        args.append("--no-run")
    if text:
        args.append("--text")
    elif json:
        args.append("--json")
    else:
        if out is None:
            out = os.path.splitext(os.path.basename(p))[0] + ".atlas.html"
        out = os.path.abspath(str(out))
        args += ["-o", out]
    try:
        r = subprocess.run(args, capture_output=True, text=True, env=_core._env(), timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError("viv atlas %s: still running after %g s; run=False skips the run" % (path, timeout))
    if r.returncode != 0:
        raise RuntimeError("viv atlas %s: %s" % (path, r.stderr.strip()))
    if text:
        return r.stdout
    if json:
        return _json.loads(r.stdout)
    return out
