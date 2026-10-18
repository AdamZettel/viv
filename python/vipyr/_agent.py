# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The vipyr front door for agents: `vipyr catalog`,
`vipyr help [--agent [--brief]]`, and the viv commands passed through to the
bundled compiler, so an agent starts with one name.

`vipyr catalog` reads engines.json beside this file: each validated engine,
what it solves, the reference it was checked against, its claims, its code
map and, where one was measured, the comparison with plain Python. The
numbers in it are copied from the committed files each entry names.

With --json, catalog prints the same envelope viv prints:

    {"viv": 1, "cmd": "catalog", "ok": true, "exit": 0, "diagnostics": [],
     "result": {...}, "truncated": false, "more": null}
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from . import _core

ENGINES = Path(__file__).with_name("engines.json")

#: exit codes (viv help exit-codes)
OK, USAGE, INTERNAL = 0, 2, 70
#: what --summary keeps of a list, and --max-lines' default
SUMMARY_ITEMS, DEFAULT_MAX_LINES = 5, 200

#: the viv commands vipyr passes through unchanged
PASS_THROUGH = ("run", "check", "features", "explain", "map", "claims", "fix", "digits", "skill", "hook",
                "whatis", "lsfunc", "src", "lint")

HELP = """vipyr: the front door to Viv (`import vipyr as vp` in Python; `viv` is the language's tool).

  vipyr catalog [--topic T] [--json] [--summary]   the validated engines: what each solves, its
                                                   reference and agreement, claims, code map, and a
                                                   timings measured beside plain Python where they exist
  vipyr help --agent [--brief]                     the working guide for an agent (catalog first)
  vipyr run|check|features|explain|map|claims ...  the viv commands, through the bundled compiler
  vipyr skill install [--project|--user|--plugin]  the Claude Code skill
  vipyr stubs FILE.viv ...                         write FILE.pyi beside each source
  vipyr build FILE.viv [--fast]                    compile into the cache; print the entry
  vipyr compile model.pkl -o m.c                   a fitted scikit-learn pipeline as one C99 file
  vipyr compiler                                   the viv binary and Viv's library root
  vipyr viv ARGS...                                run the bundled viv binary

viv help lists the language's commands; viv help json describes the envelope."""


def envelope(cmd, exit_code, result, diagnostics=(), truncated=False, more=None):
    return {"viv": 1, "cmd": cmd, "ok": exit_code == OK, "exit": exit_code, "diagnostics": list(diagnostics),
            "result": result, "truncated": truncated, "more": more}


def diag(message, rule):
    return {"severity": "error", "rule": rule, "message": message, "file": None, "line": None, "col": None,
            "notes": []}


def _flag_value(argv, name):
    if name in argv:
        i = argv.index(name)
        if i + 1 >= len(argv):
            raise ValueError("%s takes a value" % name)
        return argv[i + 1]
    return None


def load_engines():
    return json.loads(ENGINES.read_text())


def catalog(argv):
    """`vipyr catalog [--topic T] [--json] [--summary] [--max-lines N]`."""
    as_json = "--json" in argv
    try:
        topic = _flag_value(argv, "--topic")
        max_lines = int(_flag_value(argv, "--max-lines") or DEFAULT_MAX_LINES)
    except ValueError as e:
        return _fail("catalog", str(e), "U-usage", as_json)
    summary = "--summary" in argv
    data = load_engines()
    engines = data["engines"]
    topics = sorted({e["topic"] for e in engines})
    if topic:
        if topic not in topics:
            return _fail("catalog", "no topic %r; the topics are %s" % (topic, ", ".join(topics)), "U-usage", as_json)
        engines = [e for e in engines if e["topic"] == topic]
    keep = SUMMARY_ITEMS if summary else max_lines
    cut = keep > 0 and len(engines) > keep and not summary
    if summary:
        shown = [dict({k: e.get(k) for k in ("id", "topic", "solves", "vipyr", "viv")},
                      comparison=(e.get("comparison") or {}).get("speed"),
                      viv_slower=(e.get("comparison") or {}).get("viv_slower")) for e in engines]
    else:
        shown = engines[:keep] if cut else engines
    if as_json:
        result = {"note": data.get("note", ""), "topics": topics, "count": len(engines), "engines": shown}
        more = "--max-lines 0 for every engine" if cut else None
        print(json.dumps(envelope("catalog", OK, result, truncated=cut, more=more)))
        return OK
    print(catalog_text(shown if not summary else engines, data.get("note", "")))
    return OK


def catalog_text(engines, note=""):
    out = []
    for e in engines:
        out.append("%-24s %-9s %s" % (e["id"], e["topic"], e["solves"]))
        out.append("%-24s   checked: %s -- %s" % ("", e.get("reference", ""), e.get("agreement", "")))
        c = e.get("comparison")
        if c and c.get("speed"):
            machine = " [%s]" % c["machine"] if c.get("machine") else ""
            out.append("%-24s   measured with %s: %s%s" % ("", c.get("against", ""), c["speed"], machine))
        if c and c.get("lines"):
            out.append("%-24s   lines: %s" % ("", c["lines"]))
        use = e.get("vipyr") or e.get("viv")
        if use:
            out.append("%-24s   use: %s" % ("", use))
    if note:
        out.append("")
        out.append(note)
    return "\n".join(out)


def _fail(cmd, message, rule, as_json):
    code = USAGE if rule.startswith("U-") else INTERNAL
    if as_json:
        print(json.dumps(envelope(cmd, code, None, [diag(message, rule)])))
    else:
        print("vipyr %s: %s" % (cmd, message), file=sys.stderr)
    return code


def viv(argv):
    """Run the bundled viv with these arguments; its exit code."""
    return subprocess.call([_core.compiler()] + list(argv), env=_core._env())


def help_(argv):
    """`vipyr help [--agent [--brief]] [<command>]`: --agent is viv's guide,
    which leads with the catalog (the same engines.json)."""
    if not argv:
        print(HELP)
        return OK
    return viv(["help"] + list(argv))
