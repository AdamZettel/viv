#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Checks of tools/vivdb.

    python3 tools/vivdb/test_vivdb.py            the fixture (fast; the premerge check `vivdb`)
    python3 tools/vivdb/test_vivdb.py --db PATH  also the consistency checks on a built database

On the fixture (tools/vivdb/fixture): build a database, check that every
stored file's bytes are the file's bytes and hash to its git blob id, that
every def and test span lies inside its file and starts where the source
says, that a second build changes no row, and that the saved `impact` query
for `exp` finds the fixture's callers, program and test.
"""

import hashlib
import os
import sqlite3
import subprocess
import sys
import tempfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vivdb  # noqa: E402

ROOT = vivdb.repo_root()
FIXTURE = "tools/vivdb/fixture"


def git_blob(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def check_consistency(db):
    """Every module's stored bytes are its blob; every span is valid."""
    con = sqlite3.connect(db)
    bad = []
    n_defs = n_mods = 0
    for module, path, blob in con.execute("SELECT module, path, blob FROM modules"):
        n_mods += 1
        data = vivdb.file_bytes(con, module)
        if data is None or git_blob(data) != blob:
            bad.append(f"{path}: stored bytes do not hash to blob {blob[:8]}")
            continue
        for name, b0, b1, line in con.execute("SELECT name, byte_start, byte_end, line_start FROM def_sites WHERE module = ? AND kind IN ('def', 'closure')", (module,)):
            n_defs += 1
            if not (0 <= b0 < b1 <= len(data)):
                bad.append(f"{path}: {name}: span {b0}..{b1} outside {len(data)} bytes")
                continue
            text = data[b0:b1].decode("utf-8", "replace")
            at = data[:b0].count(b"\n") + 1
            if at != line:
                bad.append(f"{path}: {name}: span starts on line {at}, facts say {line}")
            if name.startswith("viv__fnbody") and not text.startswith("fn"):
                bad.append(f"{path}: {name}: span text {text[:40]!r} is not the closure")
            elif not name.startswith("viv__fnbody") and name not in text.split("(", 1)[0] and not text.lstrip().startswith(("def", "placeholder")):
                bad.append(f"{path}: {name}: span text {text[:40]!r} is not the def")
        for name, b0, b1 in con.execute("SELECT name, byte_start, byte_end FROM tests WHERE module = ?", (module,)):
            if not (0 <= b0 < b1 <= len(data)) or not data[b0:b1].lstrip().startswith(b"test"):
                bad.append(f"{path}: test {name!r}: span {b0}..{b1} is not the test")
    # every call's resolved callee is a def the database has
    for cm, c, n in con.execute("""SELECT DISTINCT c.callee_module, c.callee, c.callee_name FROM calls c
                                   WHERE c.callee_module IS NOT NULL AND c.callee_module IN (SELECT module FROM modules)
                                   AND NOT EXISTS (SELECT 1 FROM def_sites s WHERE s.module = c.callee_module AND s.name = c.callee)"""):
        bad.append(f"call to {cm}.{c} ({n}): no such def")
    con.close()
    return n_mods, n_defs, bad


def dump(db):
    """Every derived row, for comparing two builds."""
    con = sqlite3.connect(db)
    out = []
    for t in vivdb.DERIVED_TABLES:
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})") if r[1] != "indexed_at"]
        rows = con.execute(f"SELECT {', '.join(cols)} FROM {t}").fetchall()
        out.append((t, sorted(map(repr, rows))))
    con.close()
    return out


def parity(db):
    """`viv db query|show` (Rust, the SQLite inside viv) against this tool:
    the same stdout for every saved query, in table and JSON form, an ad-hoc
    query and `show --as graph|intent`."""
    import contextlib
    import io
    viv = os.environ.get("VIV") or os.path.join(vivdb.repo_root(), "target", "release", "viv")
    if not os.path.exists(viv):
        return [f"parity: no viv binary at {viv}"]
    params = {"impact": ["target=exp"], "tests-covering": [f"def={FIXTURE}/fx_lib.fx_decay"],
              "builtin-callers": ["name=exp"], "feature-uses": ["feature=steps"]}
    runs = []
    for name in vivdb.saved_queries():
        ps = [x for p in params.get(name, []) for x in ("--param", p)]
        runs.append(["query", "--saved", name, "--db", db] + ps)
        runs.append(["query", "--saved", name, "--db", db, "--json"] + ps)
    runs.append(["query", "SELECT module, name, line_start, 1.5 * line_start AS x, NULL AS n FROM def_sites ORDER BY 1, 2", "--db", db])
    runs.append(["query", "SELECT name, 0.1 AS a, 1e-07 AS b, 1e20 AS c FROM def_sites ORDER BY 1", "--db", db, "--json"])
    for ref in ("fx_decay", f"{FIXTURE}/fx_lib.fx_left"):
        for as_ in ("graph", "intent"):
            runs.append(["show", ref, "--as", as_, "--db", db])
    bad = []
    for args in runs:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            try:
                vivdb.main(args)
            except SystemExit as e:
                print(f"exit {e.code}")
        got = subprocess.run([viv, "db"] + args, capture_output=True, text=True)
        out = got.stdout + ("" if got.returncode == 0 else f"exit {got.stderr.strip()}\n")
        if out != buf.getvalue():
            bad.append(f"parity: viv db {' '.join(args)}: Rust and Python differ:\n--- python\n{buf.getvalue()}--- viv\n{out}")
    print(f"parity: {len(runs)} runs of `viv db` against vivdb.py, {len(bad)} differ")
    return bad


def main():
    fails = []
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "fx.sqlite")
        vivdb.main(["build", "--db", db, "--jobs", "2", "--paths", FIXTURE])
        first = dump(db)
        n_mods, n_defs, bad = check_consistency(db)
        fails += bad
        if n_mods != 2 or n_defs != 4:
            fails.append(f"fixture: {n_mods} modules and {n_defs} defs, expected 2 and 4")
        vivdb.main(["build", "--db", db, "--paths", FIXTURE])
        if dump(db) != first:
            fails.append("a second build changed the tables")
        # the saved impact query for exp
        con = vivdb.connect_ro(db)
        sql = open(vivdb.saved_queries()["impact"]).read()
        _, rows = vivdb.run_query(con, sql, {"target": "exp"})
        got = {(r[0], r[1]) for r in rows}
        want = {("def", f"{FIXTURE}/fx_lib.fx_decay"), ("def", f"{FIXTURE}/fx_lib.fx_left"), ("program", f"{FIXTURE}/fx_prog")}
        if got != want:
            fails.append(f"impact exp: got {sorted(got)}, expected {sorted(want)}")
        _, rows = vivdb.run_query(con, open(vivdb.saved_queries()["dead-code"]).read(), {})
        if [r[1] for r in rows] != ["fx_unused"]:
            fails.append(f"dead-code: got {rows}")
        _, rows = vivdb.run_query(con, open(vivdb.saved_queries()["tests-covering"]).read(), {"def": f"{FIXTURE}/fx_lib.fx_decay"})
        if [r[0] for r in rows] != [f"{FIXTURE}/fx_prog:half is left at the half-life"]:
            fails.append(f"tests-covering: got {rows}")
        con.close()
        fails += parity(db)
    if "--db" in sys.argv:
        full = sys.argv[sys.argv.index("--db") + 1]
        n_mods, n_defs, bad = check_consistency(full)
        print(f"{full}: {n_mods} modules, {n_defs} defs, {len(bad)} problems")
        fails += bad[:50]
    for f in fails:
        print("FAIL", f)
    print("vivdb tests:", "FAIL" if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
