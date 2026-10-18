# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vipyr's front door for agents: `vipyr catalog`
prints the envelope with the result schema lang/viv/schemas/catalog.json, and
every path an engine names is in the repository."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / "python"
SCHEMAS = ROOT / "lang/viv/schemas"


def vipyr(*args):
    r = subprocess.run([sys.executable, "-m", "vipyr", *args], cwd=PY, capture_output=True, text=True,
                       stdin=subprocess.DEVNULL)
    return r.returncode, r.stdout, r.stderr


def validate(v, s, at="$"):
    """The part of JSON Schema the schemas use."""
    kinds = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None), "integer": int}
    t = s.get("type")
    if t:
        ts = t if isinstance(t, list) else [t]
        assert any(isinstance(v, kinds[k]) and not (k == "integer" and isinstance(v, bool)) for k in ts), (at, v)
    if "enum" in s:
        assert v in s["enum"], (at, v)
    if "const" in s:
        assert v == s["const"], (at, v)
    if isinstance(v, dict):
        for r in s.get("required", []):
            assert r in v, (at, r)
        for k, x in v.items():
            if k in s.get("properties", {}):
                validate(x, s["properties"][k], "%s.%s" % (at, k))
    if isinstance(v, list) and "items" in s:
        for i, x in enumerate(v):
            validate(x, s["items"], "%s[%d]" % (at, i))


def schema(name):
    return json.loads((SCHEMAS / ("%s.json" % name)).read_text())


def test_catalog_json_is_the_envelope_with_its_schema():
    code, out, err = vipyr("catalog", "--json", "--max-lines", "0")
    assert code == 0, err
    env = json.loads(out)
    validate(env, schema("envelope"))
    validate(env["result"], schema("catalog"))
    assert env["result"]["count"] == len(env["result"]["engines"]) > 0


def test_every_path_an_engine_names_is_in_the_repository():
    tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split())
    engines = json.loads((PY / "vipyr/engines.json").read_text())["engines"]
    for e in engines:
        for p in [e.get("viv")] + e["evidence"] + e["claims"] + [(e.get("comparison") or {}).get("source")]:
            if p:
                assert p in tracked or (ROOT / p).exists(), (e["id"], p)
        c = e.get("comparison")
        if c and c.get("speed"):
            assert isinstance(c.get("viv_slower"), bool), e["id"]


def test_track_filter_and_summary():
    code, out, _ = vipyr("catalog", "--topic", "power", "--json")
    env = json.loads(out)
    assert code == 0 and env["result"]["engines"]
    assert {e["topic"] for e in env["result"]["engines"]} == {"power"}
    code, out, _ = vipyr("catalog", "--topic", "nope", "--json")
    env = json.loads(out)
    assert code == 2 and env["diagnostics"][0]["rule"] == "U-usage"
    code, out, _ = vipyr("catalog", "--json", "--summary")
    engines = json.loads(out)["result"]["engines"]
    assert all(set(e) == {"id", "topic", "solves", "vipyr", "viv", "comparison", "viv_slower"} for e in engines)


def test_help_lists_the_front_door():
    code, out, _ = vipyr("help")
    assert code == 0 and "vipyr catalog" in out
