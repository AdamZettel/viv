# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The Viv kernel for Jupyter.

Jupyter sends the kernel one cell at a time. The kernel keeps the cells it
has run, in order, and runs a new cell as the program "every earlier cell,
then this one" (see runner.py), showing only what this cell printed. So a
cell sees the definitions above it, and the notebook as a whole is one Viv
program, the same one `viv run` runs from the .viv file.

Re-running a cell: when the front end says which cell it is (JupyterLab and
VS Code send a cell id), the kernel drops that cell and every cell run after
it, then runs it again after the cells before it. That is what "running cell
N means running cells 1..N" asks for, and it keeps an edited cell from
running twice. Without an id, running the same source as the last cell again
replaces it; anything else is appended.

A comment on a cell's first line picks the engine:

    #% native      this cell runs through the C backend (compiled, fast)
    #% interp      this cell runs through the interpreter (the default)
    #% engine native / #% engine interp
                   the default for the rest of the session
    #% reset       forget every cell run so far

They are Viv comments, so a notebook that uses them still runs as a program.
"""

import os
import re

from ipykernel.kernelbase import Kernel

from . import __version__
from .figures import describe, parse, render
from .runner import Runner

_DIRECTIVE = re.compile(r"^#%\s*(.*?)\s*$")


class VivKernel(Kernel):
    implementation = "vivkernel"
    implementation_version = __version__
    language = "viv"
    language_version = "0.1"
    language_info = {
        "name": "viv",
        "mimetype": "text/x-viv",
        "file_extension": ".viv",
        # Viv has no highlighter of its own in Jupyter; Python's colors its
        # `def`, `for`, `if`, `return`, numbers and `#` comments correctly.
        "codemirror_mode": "python",
        "pygments_lexer": "python",
    }
    banner = "Viv: each cell runs after every cell above it; `#% native` compiles a cell to C"
    help_links = [{"text": "Viv", "url": "https://viv-lang.org"}]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.runner = Runner()
        self.engine = os.environ.get("VIV_ENGINE", "interp")
        # (cell id or None, source) for every cell that ran, in order
        self.history = []

    # -- directives -------------------------------------------------------
    def _directives(self, code):
        """Read `#% ...` lines at the top of the cell. Returns the engine for
        this cell and whether the cell has any code besides directives."""
        engine = self.engine
        lines = code.split("\n")
        i = 0
        while i < len(lines):
            m = _DIRECTIVE.match(lines[i].strip())
            if not m:
                break
            words = m.group(1).split()
            if words == ["native"] or words == ["interp"]:
                engine = words[0]
            elif len(words) == 2 and words[0] == "engine" and words[1] in ("native", "interp"):
                self.engine = engine = words[1]
            elif words == ["reset"]:
                self.history = []
            i += 1
        rest = "\n".join(lines[i:]).strip()
        return engine, bool(rest)

    # -- the cell's place in the notebook -----------------------------------
    def _prefix(self, code, cell_id):
        if cell_id is not None:
            for k, (cid, _) in enumerate(self.history):
                if cid == cell_id:
                    return k
        elif self.history and self.history[-1][1] == code:
            return len(self.history) - 1
        return len(self.history)

    # -- execution ----------------------------------------------------------
    def do_execute(self, code, silent, store_history=True, user_expressions=None,
                   allow_stdin=False, *, cell_id=None):
        engine, has_code = self._directives(code)
        if not has_code:
            return self._ok()
        at = self._prefix(code, cell_id)
        prefix = [src for _, src in self.history[:at]]
        try:
            res = self.runner.run(prefix, code, engine=engine)
        except KeyboardInterrupt:
            return {"status": "abort", "execution_count": self.execution_count}
        except OSError as e:
            return self._error(f"error: {e}", silent)

        if res.error:
            # the cell did not run: it is not added, and cells after it are
            # dropped, since they were built on a version of it that is gone
            del self.history[at:]
            if res.stderr and not silent:
                self._stream("stderr", res.stderr)
            return self._error(res.error, silent)

        del self.history[at:]
        self.history.append((cell_id, code))
        if not silent:
            self._show(res.output)
            if res.stderr:
                self._stream("stderr", res.stderr)
        return self._ok()

    def _ok(self):
        return {"status": "ok", "execution_count": self.execution_count,
                "payload": [], "user_expressions": {}}

    def _error(self, message, silent):
        first = message.split("\n", 1)[0]
        evalue = first[len("error:"):].strip() if first.startswith("error:") else first
        if not silent:
            self.send_response(self.iopub_socket, "error", {
                "ename": "VivError", "evalue": evalue, "traceback": message.split("\n"),
            })
        return {"status": "error", "execution_count": self.execution_count,
                "ename": "VivError", "evalue": evalue, "traceback": message.split("\n")}

    def _stream(self, name, text):
        self.send_response(self.iopub_socket, "stream", {"name": name, "text": text + "\n"})

    def _show(self, output):
        """Send the cell's output in order: text as a stream, a failed test's
        FAIL line on stderr (front ends color it), figures as SVG."""
        for kind, value in parse(output):
            if kind == "text":
                buf = []
                for line in value.split("\n"):
                    if line.startswith("FAIL "):
                        if buf:
                            self._stream("stdout", "\n".join(buf))
                            buf = []
                        self._stream("stderr", line)
                    else:
                        buf.append(line)
                if buf:
                    self._stream("stdout", "\n".join(buf))
            elif kind == "narrate":
                self.send_response(self.iopub_socket, "display_data", {
                    "data": {"text/plain": value, "text/markdown": value}, "metadata": {},
                })
            elif kind == "note":
                self._stream("stdout", f"[{value}]")
            else:
                self.send_response(self.iopub_socket, "display_data", {
                    "data": {"image/svg+xml": render(value), "text/plain": describe(value)},
                    "metadata": {},
                })

    def do_shutdown(self, restart):
        self.runner.close()
        return {"status": "ok", "restart": restart}

    def do_is_complete(self, code):
        # a block is open while braces or brackets are unbalanced
        depth = 0
        for line in code.split("\n"):
            line = line.split("#", 1)[0]
            depth += line.count("{") + line.count("(") + line.count("[")
            depth -= line.count("}") + line.count(")") + line.count("]")
        return {"status": "incomplete" if depth > 0 else "complete", "indent": "    " if depth > 0 else ""}
