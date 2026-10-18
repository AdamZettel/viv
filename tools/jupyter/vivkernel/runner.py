# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Run one notebook cell of a Viv program and return only that cell's output.

A Viv notebook cell is not run on its own: running cell N runs cells 1..N as
one program, so the cell sees every definition above it exactly as it would
when the whole file runs. To show only the NEW output, the program gets a
marker statement between the earlier cells and this one,

    __cell_boundary__ = 0.0
    show __cell_boundary__

and everything the program prints after the marker's line is this cell's
output. The site's in-browser cells (www/src/components/VivCell.jsx) cut
their output with the same marker, so a cell prints the same thing in
Jupyter and on the page.

Two engines run a cell. The interpreter (`viv run`) is the default: it
starts at once. The C backend (`viv prog.viv -o prog.c`, then cc) costs a
compile but runs loops at native speed, for cells that do real work. Both
engines print the same output by design; tools/both-engines.sh checks that.

This module has no Jupyter dependency, so the tests can call it directly.
"""

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

MARKER_NAME = "__cell_boundary__"
MARKER_STMT = f"{MARKER_NAME} = 0.0\nshow {MARKER_NAME}"
MARKER_LINE = f"{MARKER_NAME}: 0"

# Summary lines the engines add at the end of a run with failed tests. The
# FAIL lines themselves are the cell's output; the count adds nothing.
_SUMMARY = re.compile(r"^\d+ (assertion|hold)s? FAILED$")
# Interpreter diagnostics that are data for the site's lenses, not output.
_DIAGNOSTIC = ("__graph__", "__trace__", "__steps__", "__story__")


def find_viv():
    """The viv binary: $VIV_BIN, else this repository's release build, else PATH."""
    env = os.environ.get("VIV_BIN")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    repo_bin = os.path.normpath(os.path.join(here, "..", "..", "..", "target", "release", "viv"))
    if os.path.exists(repo_bin):
        return repo_bin
    found = shutil.which("viv")
    if found:
        return found
    raise FileNotFoundError(
        "no viv binary: build it with `cargo build --release -p viv`, or set VIV_BIN"
    )


@dataclass
class CellResult:
    output: str = ""  # what the cell printed, marker and summaries removed
    error: str = ""  # the engine's diagnostic, empty when the cell ran
    stderr: str = ""  # anything else the engine wrote to stderr
    engine: str = "interp"
    seconds: float = 0.0
    failed_tests: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.error


def build_program(prefix_cells, cell):
    """The source that runs `cell` after `prefix_cells`, and the 1-based line
    where the cell's own source starts in it (for mapping error locations)."""
    if not prefix_cells:
        return cell, 1
    head = "\n".join(prefix_cells) + "\n" + MARKER_STMT + "\n"
    return head + cell, head.count("\n") + 1


def cut_at_marker(text, has_prefix):
    """The part of `text` printed after the marker line."""
    if not has_prefix:
        return text
    lines = text.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if lines[i] == MARKER_LINE:
            return "\n".join(lines[i + 1:])
    # no marker: the program failed before reaching this cell
    return ""


def _clean(text):
    keep = []
    for line in text.split("\n"):
        if line.startswith(_DIAGNOSTIC) or _SUMMARY.match(line):
            continue
        keep.append(line)
    return "\n".join(keep)


class Runner:
    """Runs cells with one viv binary. `workdir` is where the program file is
    written, so that an import finds modules beside the notebook."""

    def __init__(self, viv=None, workdir=None, cc=None, timeout=None):
        self.viv = viv or find_viv()
        self.workdir = workdir or os.getcwd()
        self.cc = cc or os.environ.get("CC", "cc")
        self.timeout = timeout
        self._scratch = tempfile.mkdtemp(prefix="vivkernel-")
        # compiled programs by source hash, so re-running a native cell whose
        # prefix has not changed does not compile again
        self._binaries = {}
        self.current = None  # the running subprocess, for interrupts

    def close(self):
        shutil.rmtree(self._scratch, ignore_errors=True)

    # -- the program file -------------------------------------------------
    def _write_program(self, source):
        # Beside the notebook when that directory is writable (module lookup
        # is relative to the program's own directory), else in scratch.
        for d in (self.workdir, self._scratch):
            try:
                fd, path = tempfile.mkstemp(prefix=".vivcell-", suffix=".viv", dir=d)
            except OSError:
                continue
            with os.fdopen(fd, "w") as f:
                f.write(source)
            return path
        raise OSError("cannot write the cell's program anywhere")

    def _exec(self, argv):
        self.current = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=self.workdir
        )
        try:
            out, err = self.current.communicate(timeout=self.timeout)
        except BaseException:
            # an interrupt or a timeout: stop the engine, then pass it on
            self.current.kill()
            self.current.communicate()
            raise
        finally:
            code = self.current.returncode
            self.current = None
        return code, out, err

    # -- running ----------------------------------------------------------
    def run(self, prefix_cells, cell, engine="interp"):
        import time

        source, first_line = build_program(prefix_cells, cell)
        t0 = time.perf_counter()
        path = self._write_program(source)
        try:
            if engine == "native":
                res = self._run_native(path, source)
            else:
                res = self._run_interp(path)
        finally:
            os.unlink(path)
        res.engine = engine
        res.seconds = time.perf_counter() - t0
        if res.error:
            res.error = self._located(path, source, first_line, res.error)
        else:
            res.output = cut_at_marker(_clean(res.output), bool(prefix_cells)).strip("\n")
            res.failed_tests = [l[5:] for l in res.output.split("\n") if l.startswith("FAIL ")]
        res.stderr = _clean(res.stderr).strip("\n")
        return res

    def _run_interp(self, path):
        code, out, err = self._exec([self.viv, "run", "--quiet", path])
        res = CellResult(output=out)
        errors = [l for l in err.split("\n") if l.startswith("error")]
        if errors:
            res.error = err.strip("\n")
        else:
            res.stderr = err
        return res

    def _run_native(self, path, source):
        key = hashlib.sha256(source.encode()).hexdigest()[:16]
        exe = self._binaries.get(key)
        if exe is None:
            c_file = os.path.join(self._scratch, key + ".c")
            exe = os.path.join(self._scratch, key)
            code, out, err = self._exec([self.viv, path, "-o", c_file])
            if code != 0:
                return CellResult(error=(err or out).strip("\n"))
            code, out, err = self._exec([self.cc, "-O2", "-o", exe, c_file, "-lm"])
            if code != 0:
                first = [l for l in err.split("\n") if "error" in l][:3]
                return CellResult(error="error: the C compiler rejected the generated program\n" + "\n".join(first))
            self._binaries[key] = exe
        code, out, err = self._exec([exe])
        res = CellResult(output=out)
        if code < 0:
            res.error = f"error: the native program stopped on signal {-code}"
        errors = [l for l in err.split("\n") if l.startswith("error")]
        if errors:
            res.error = err.strip("\n")
        else:
            res.stderr = err
        return res

    # -- error locations --------------------------------------------------
    def _located(self, path, source, first_line, message):
        """Point a diagnostic at the cell's own lines.

        The interpreter's messages carry no location (`error: undefined
        variable x`). The compiler front end reports the same mistakes with
        a file, line and caret, so when the interpreter fails the program is
        checked once more with it, and a located message replaces the bare
        one. Line numbers are rewritten from the combined program to the
        cell, since the reader sees the cell, not the program."""
        if "-->" not in message:
            fd, again = tempfile.mkstemp(suffix=".viv", dir=self._scratch)
            with os.fdopen(fd, "w") as f:
                f.write(source)
            try:
                _, out, err = self._exec([self.viv, again, "-o", again[:-4] + ".c"])
            finally:
                os.unlink(again)
                if os.path.exists(again[:-4] + ".c"):
                    os.unlink(again[:-4] + ".c")
            located = err.strip("\n")
            if "-->" in located:
                message = located
                path = again
        return relocate(message, path, first_line)


def relocate(message, path, first_line):
    """Rewrite `--> <path>:L:C` and the `  L | source` gutter so L counts
    from the cell's first line. A location before the cell is named as an
    earlier cell's."""
    name = re.escape(path)
    out = []
    for line in message.split("\n"):
        m = re.match(rf"^(\s*-->\s*){name}:(\d+):(\d+)(.*)$", line)
        if m:
            n = int(m.group(2))
            if n >= first_line:
                line = f"{m.group(1)}this cell, line {n - first_line + 1}, column {m.group(3)}{m.group(4)}"
            else:
                line = f"{m.group(1)}an earlier cell (line {n} of the notebook's code so far), column {m.group(3)}{m.group(4)}"
        else:
            g = re.match(r"^(\s*)(\d+)( \|.*)$", line)
            if g and int(g.group(2)) >= first_line:
                # keep the gutter's width so the caret line below still lines up
                n = str(int(g.group(2)) - first_line + 1).rjust(len(g.group(2)))
                line = f"{g.group(1)}{n}{g.group(3)}"
        out.append(line)
    return "\n".join(out)
