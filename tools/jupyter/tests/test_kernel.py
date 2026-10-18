# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The kernel under Jupyter: install the spec into a scratch prefix, execute
the example notebooks with `jupyter nbconvert --execute`, and check each
cell's outputs. Nothing is installed for the user."""

import json
import os
import queue
import shutil
import subprocess
import sys

import pytest

import vivnb

HERE = os.path.dirname(os.path.abspath(__file__))
JUPYTER = os.path.dirname(HERE)
EXAMPLES = os.path.join(JUPYTER, "examples")

pytestmark = pytest.mark.skipif(shutil.which("jupyter") is None, reason="jupyter is not installed")


@pytest.fixture(scope="module")
def jupyter_env(tmp_path_factory):
    prefix = tmp_path_factory.mktemp("prefix")
    subprocess.run([sys.executable, os.path.join(JUPYTER, "install.py"), "--prefix", str(prefix)],
                   check=True, capture_output=True)
    env = dict(os.environ)
    env["JUPYTER_PATH"] = os.path.join(str(prefix), "share", "jupyter")
    env["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
    return env


def execute(env, viv_path, tmp_path, allow_errors=False):
    """Convert a .viv to .ipynb, run it with nbconvert and the viv kernel, return the result."""
    name = os.path.splitext(os.path.basename(viv_path))[0]
    nb_path = tmp_path / f"{name}.ipynb"
    with open(viv_path) as f:
        vivnb.write_ipynb(vivnb.to_ipynb(f.read(), os.path.basename(viv_path)), str(nb_path))
    cmd = ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--ExecutePreprocessor.kernel_name=viv",
           "--ExecutePreprocessor.timeout=120", str(nb_path), "--output", f"{name}.out.ipynb"]
    if allow_errors:
        cmd.append("--allow-errors")
    subprocess.run(cmd, check=True, capture_output=True, env=env, cwd=str(tmp_path))
    return json.loads((tmp_path / f"{name}.out.ipynb").read_text())


def outputs(nb):
    """For each code cell: (stdout text, stderr text, number of SVG figures, errors)."""
    out = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        stdout = "".join("".join(o["text"]) for o in c["outputs"] if o["output_type"] == "stream" and o["name"] == "stdout")
        stderr = "".join("".join(o["text"]) for o in c["outputs"] if o["output_type"] == "stream" and o["name"] == "stderr")
        svgs = [o for o in c["outputs"] if o["output_type"] == "display_data" and "image/svg+xml" in o["data"]]
        errors = [o for o in c["outputs"] if o["output_type"] == "error"]
        out.append((stdout, stderr, len(svgs), errors))
    return out


EXPECTED = {
    "heat-rod.viv": [
        ("", 1),
        ("", 0),
        ("t_end: 0.125\nmid: 0.290954\nexact_mid: 0.291213\nPASS the middle cools at the exact rate, within 0.1%\n", 1),
        ("largest: 4.2007e+26\nPASS r = 0.6 has blown up: some point is above the starting maximum\n"
         "PASS r = 0.4 stays below the starting maximum everywhere\n", 1),
    ],
    "newton-vs-bisection.viv": [
        ("", 0),
        ("bis_steps: 37\n", 0),
        ("new_steps: 5\nguess: 1.41421\nPASS bisection needs 37 steps to reach 1e-12\nPASS Newton needs 5\n"
         "PASS Newton's third step has about twice the correct digits of its second\n", 2),
    ],
    "sums-of-uniforms.viv": [
        ("", 0),
        ("mean: 6.00161\nvariance: 0.998062\nworst: 0.0144551\nPASS the mean is 6 and the variance is 1, to 0.02\n"
         "PASS every bar is within 0.02 of the normal curve\n", 1),
    ],
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_example_notebook_executes(jupyter_env, tmp_path, name):
    nb = execute(jupyter_env, os.path.join(EXAMPLES, name), tmp_path)
    got = outputs(nb)
    assert len(got) == len(EXPECTED[name])
    for (stdout, stderr, svgs, errors), (want_text, want_svgs) in zip(got, EXPECTED[name]):
        assert errors == []
        assert stderr == ""
        assert stdout == want_text
        assert svgs == want_svgs
    # no program file is left beside the notebook
    assert not [p for p in os.listdir(tmp_path) if p.startswith(".vivcell-")]


def test_failing_test_and_compile_error(jupyter_env, tmp_path):
    src = tmp_path / "broken.viv"
    src.write_text(
        "###\n## A test that fails\n###\n\n"
        "x = 2.0\nshow x\ntest \"x is three\" {\n    assert_near(x, 3.0, 1.0e-12)\n}\n\n"
        "###\n## A name that does not exist\n###\n\n"
        "y = 1.0\nshow undefined_name\n\n"
        "###\n## Later cells still run\n###\n\n"
        "show x + 1.0\n"
    )
    nb = execute(jupyter_env, str(src), tmp_path, allow_errors=True)
    (out1, err1, _, e1), (out2, _, _, e2), (out3, _, _, e3) = outputs(nb)
    assert out1 == "x: 2\n"
    assert err1 == "FAIL x is three -- assert_near: |2 - 3| = 1 > 1e-12\n"
    assert e1 == []
    assert len(e2) == 1 and e2[0]["ename"] == "VivError"
    assert "unknown variable `undefined_name`" in e2[0]["evalue"]
    assert "  --> this cell, line 2, column 6" in e2[0]["traceback"]
    # the cell that failed is not part of the program the next cell runs after
    assert e3 == [] and out3 == "3\n"


def test_rerunning_a_cell_by_id_replaces_it(jupyter_env):
    """What JupyterLab and VS Code do: they send a cell id, and running a cell
    again (edited) drops it and every cell after it before running."""
    from jupyter_client.manager import start_new_kernel

    old = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = jupyter_env["JUPYTER_PATH"]
    try:
        km, kc = start_new_kernel(kernel_name="viv")
    finally:
        if old is None:
            del os.environ["JUPYTER_PATH"]
        else:
            os.environ["JUPYTER_PATH"] = old

    def run(code, cell_id):
        msg_id = kc.execute(code, cell_id=cell_id) if "cell_id" in kc.execute.__code__.co_varnames else None
        if msg_id is None:
            # older jupyter_client: put the id where JupyterLab puts it
            msg = kc.session.msg("execute_request", {"code": code, "silent": False, "store_history": True,
                                                     "user_expressions": {}, "allow_stdin": False,
                                                     "stop_on_error": True},
                                 metadata={"cellId": cell_id})
            kc.shell_channel.send(msg)
            msg_id = msg["header"]["msg_id"]
        text = []
        while True:
            try:
                m = kc.get_iopub_msg(timeout=30)
            except queue.Empty:
                break
            if m["parent_header"].get("msg_id") != msg_id:
                continue
            if m["msg_type"] == "stream":
                text.append(m["content"]["text"])
            if m["msg_type"] == "error":
                text.append("ERROR " + m["content"]["evalue"])
            if m["msg_type"] == "status" and m["content"]["execution_state"] == "idle":
                break
        return "".join(text)

    try:
        assert run("a = 1.0", "c1") == ""
        assert run("b = a + 1.0\nshow b", "c2") == "b: 2\n"
        # edit the first cell: the second is dropped, so running it again
        # sees the new a, and a name it defined is gone until it runs
        assert run("a = 10.0", "c1") == ""
        assert run("show b", "c3").startswith("ERROR")
        assert run("b = a + 1.0\nshow b", "c2") == "b: 11\n"
        # the engine switch: the same cell through the C backend
        assert run("#% native\nshow b * 2.0", "c4") == "22\n"
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
