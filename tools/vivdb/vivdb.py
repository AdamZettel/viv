#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vivdb: the repository's Viv programs as one SQLite database.

    python3 tools/vivdb/vivdb.py build [--db PATH] [--viv BIN] [--jobs N] [--paths P ...]
    python3 tools/vivdb/vivdb.py query "<SQL>" [--json] [--param k=v ...]
    python3 tools/vivdb/vivdb.py query --saved NAME [--json] [--param k=v ...]
    python3 tools/vivdb/vivdb.py queries                  list the saved queries
    python3 tools/vivdb/vivdb.py show <def> --as code|graph|intent

The database is a cache: `target/vivdb/viv.sqlite`, rebuilt from the
repository at any time and never committed. Only the saved queries
(tools/vivdb/queries/*.sql) are committed.

`build` runs `viv facts --json` on every tracked .viv file whose bytes, or
whose imported modules' bytes, changed since the last build, and keeps each
file's facts in the `facts_cache` table. The tables the queries read are
then rebuilt from that cache in one transaction, together with the
repository sources that are not Viv: intent state (from the facts), the bug
ledger, the export manifests and their exact hashes, generated artifacts
beside their programs, the posts' claims and the gate history.

Defs are identified by `ast_hash` (vivhash1, design section 3.3). Calls are
keyed by the caller's site (module, name), because two modules can hold
defs with the same body and so the same `ast_hash`.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import tomllib
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
FACTS_VERSION = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS facts_cache (
  path TEXT PRIMARY KEY, blob TEXT NOT NULL, deps TEXT NOT NULL, json BLOB NOT NULL);  -- zlib
CREATE TABLE IF NOT EXISTS blobs (blob TEXT PRIMARY KEY, bytes BLOB NOT NULL);           -- zlib
"""

# The tables rebuilt from the cache on every build (text and repo rows).
DERIVED = """
CREATE TABLE files (path TEXT PRIMARY KEY, blob TEXT NOT NULL, kind TEXT NOT NULL, indexed_at TEXT NOT NULL);
CREATE TABLE modules (module TEXT PRIMARY KEY, path TEXT NOT NULL, blob TEXT NOT NULL,
  intent_path TEXT, parse_error TEXT, hash_error TEXT);
CREATE TABLE imports (module TEXT, imports TEXT, PRIMARY KEY (module, imports));
CREATE TABLE uses (module TEXT, used TEXT, PRIMARY KEY (module, used));
CREATE TABLE defs (ast_hash TEXT PRIMARY KEY, kind TEXT NOT NULL, params INTEGER, signature TEXT);
CREATE TABLE def_sites (ast_hash TEXT, module TEXT, name TEXT, kind TEXT,
  line_start INTEGER, line_end INTEGER, byte_start INTEGER, byte_end INTEGER,
  ir_hash TEXT, closure_hash TEXT, PRIMARY KEY (module, name));
CREATE TABLE params (module TEXT, def_name TEXT, ord INTEGER, name TEXT, type TEXT);
CREATE TABLE calls (caller_module TEXT, caller TEXT, caller_kind TEXT,
  callee_name TEXT, callee TEXT, callee_module TEXT, target TEXT,
  builtin INTEGER NOT NULL DEFAULT 0, value INTEGER NOT NULL DEFAULT 0, line INTEGER);
CREATE INDEX calls_callee ON calls (callee_module, callee);
CREATE INDEX calls_builtin ON calls (builtin, callee);
CREATE TABLE steps (module TEXT, def_name TEXT, ord INTEGER, step TEXT, heading TEXT,
  depth INTEGER, line_start INTEGER, line_end INTEGER);
CREATE TABLE docs (module TEXT, def_name TEXT, label TEXT, text TEXT, line INTEGER);
CREATE TABLE intent (module TEXT, def_name TEXT, step_name TEXT, ast_hash TEXT, stamp TEXT,
  state TEXT NOT NULL, source TEXT NOT NULL, PRIMARY KEY (module, def_name, step_name));
CREATE TABLE tests (test_id TEXT PRIMARY KEY, module TEXT, name TEXT, line INTEGER,
  byte_start INTEGER, byte_end INTEGER, ast_hash TEXT);
CREATE TABLE test_covers (test_id TEXT, module TEXT, name TEXT, how TEXT,
  PRIMARY KEY (test_id, module, name, how));
CREATE TABLE feature_uses (module TEXT, feature TEXT, count INTEGER, PRIMARY KEY (module, feature));
CREATE TABLE handwritten (module TEXT, pattern TEXT, feature TEXT, line INTEGER);
CREATE TABLE bugs (id TEXT PRIMARY KEY, title TEXT, status TEXT, found TEXT, fixed TEXT,
  found_by TEXT, engines TEXT, disagreement INTEGER, commit_sha TEXT, test TEXT, root_cause TEXT);
CREATE TABLE bug_files (bug_id TEXT, path TEXT, module TEXT);
CREATE TABLE exports (name TEXT PRIMARY KEY, manifest TEXT, about TEXT, files TEXT);
CREATE TABLE export_files (export TEXT, path TEXT, module TEXT, role TEXT);
CREATE TABLE export_kernels (export TEXT, case_name TEXT, module TEXT, def_name TEXT, PRIMARY KEY (export, case_name));
CREATE TABLE export_hashes (export TEXT, case_name TEXT, exact_hash TEXT, PRIMARY KEY (export, case_name));
CREATE TABLE artifacts (path TEXT, module TEXT, kind TEXT, PRIMARY KEY (path, module));
CREATE TABLE posts (slug TEXT PRIMARY KEY, module TEXT, title TEXT, kind TEXT, lane TEXT, author TEXT);
CREATE TABLE claims (claim_id TEXT PRIMARY KEY, post TEXT, module TEXT, line INTEGER, raw TEXT,
  status TEXT, producer_kind TEXT, producer_ref TEXT);
CREATE TABLE gate_runs (gate_id INTEGER PRIMARY KEY, time TEXT, tier TEXT, verdict TEXT,
  seconds REAL, branch TEXT, sha TEXT, dirty INTEGER, source TEXT);
CREATE TABLE gate_checks (gate_id INTEGER, check_name TEXT, verdict TEXT);
"""
DERIVED_TABLES = re.findall(r"CREATE TABLE (\w+)", DERIVED)
DERIVED_INDEXES = re.findall(r"CREATE INDEX (\w+)", DERIVED)


def git(root, *args, check=True):
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, check=check).stdout


def repo_root():
    return git(os.getcwd(), "rev-parse", "--show-toplevel").strip()


def default_db(root):
    return os.path.join(root, "target", "vivdb", "viv.sqlite")


def tracked(root, paths=None):
    """{path: blob} for tracked files, with the working tree's bytes for the
    files modified there (`git hash-object`)."""
    out = {}
    for line in git(root, "ls-files", "-s", "-z", "--", *(paths or [])).split("\0"):
        if not line:
            continue
        meta, path = line.split("\t", 1)
        out[path] = meta.split()[1]
    changed = [p for p in git(root, "diff", "--name-only", "-z").split("\0") if p in out]
    if changed:
        existing = [p for p in changed if os.path.exists(os.path.join(root, p))]
        for p in changed:
            if p not in existing:
                del out[p]
        if existing:
            blobs = git(root, "hash-object", "--", *existing).split()
            out.update(zip(existing, blobs))
    return out


def module_of(path):
    return path[:-4] if path.endswith(".viv") else path


def unz(b):
    return zlib.decompress(b).decode() if isinstance(b, bytes) else b


def file_bytes(con, module):
    """A module's bytes as the database holds them."""
    row = con.execute("SELECT b.bytes FROM modules m JOIN blobs b ON b.blob = m.blob WHERE m.module = ?", (module,)).fetchone()
    return zlib.decompress(row[0]) if row else None


def digest(items):
    return hashlib.sha256("\n".join(items).encode()).hexdigest()


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def run_facts(viv, root, path):
    p = subprocess.run([viv, "facts", "--json", path], cwd=root, capture_output=True, text=True, timeout=300)
    text = p.stdout.strip()
    if p.returncode != 0 or not text.startswith("{"):
        return json.dumps({"vivfacts": FACTS_VERSION, "path": path, "module": module_of(path),
                           "parse_error": (p.stderr.strip() or "viv facts failed")[:2000]})
    return text


def norm_module(m, path):
    """A module name as a path from the repository root: a relative import
    (`./fx_lib`) is resolved against the importing file's directory."""
    if m and (m.startswith("./") or m.startswith("../")):
        return os.path.normpath(os.path.join(os.path.dirname(path), m))
    return m


def deps_key(facts, blobs):
    used = [norm_module(m, facts.get("path", "")) for m in facts.get("used_modules") or []]
    return digest([f"{m}={blobs.get(m + '.viv', '-')}" for m in sorted(used)] + [f"v{FACTS_VERSION}"])


def build(args):
    root = repo_root()
    db = args.db or default_db(root)
    os.makedirs(os.path.dirname(db), exist_ok=True)
    lock = db + ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        sys.exit(f"vivdb: another build holds {lock} (remove it if no build is running)")
    try:
        return _build(args, root, db)
    finally:
        os.unlink(lock)


def _build(args, root, db):
    t0 = time.time()
    viv = args.viv or os.path.join(root, "target", "release", "viv")
    if not os.path.exists(viv):
        sys.exit(f"vivdb: no compiler at {viv}; build it (cargo build --release -p viv) or pass --viv")
    con = sqlite3.connect(db)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    blobs = tracked(root)
    viv_paths = sorted(p for p in blobs if p.endswith(".viv") and (not args.paths or any(p == q or p.startswith(q.rstrip("/") + "/") for q in args.paths)))
    cache = {r[0]: (r[1], r[2], r[3]) for r in con.execute("SELECT path, blob, deps, json FROM facts_cache")}
    cache = {p: (b, d, unz(j)) for p, (b, d, j) in cache.items()}
    todo = []
    for p in viv_paths:
        c = cache.get(p)
        if c and c[0] == blobs[p] and c[1] == deps_key(json.loads(c[2]), blobs):
            continue
        todo.append(p)
    print(f"vivdb: {len(viv_paths)} .viv files, {len(todo)} to read with `viv facts`", file=sys.stderr)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_facts, viv, root, p): p for p in todo}
        for f in concurrent.futures.as_completed(futs):
            p = futs[f]
            text = f.result()
            facts = json.loads(text)
            with open(os.path.join(root, p), "rb") as fh:
                data = fh.read()
            con.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?)", (blobs[p], zlib.compress(data, 6)))
            con.execute("INSERT OR REPLACE INTO facts_cache VALUES (?, ?, ?, ?)", (p, blobs[p], deps_key(facts, blobs), zlib.compress(text.encode(), 6)))
            done += 1
            if done % 100 == 0:
                con.commit()
                print(f"  {done}/{len(todo)}", file=sys.stderr)
    # files that left the tree leave the cache
    keep = set(viv_paths)
    for p in list(cache):
        if p not in keep:
            con.execute("DELETE FROM facts_cache WHERE path = ?", (p,))
    con.execute("DELETE FROM blobs WHERE blob NOT IN (SELECT blob FROM facts_cache)")
    con.commit()
    t1 = time.time()
    rebuild_derived(con, root, blobs)
    head = git(root, "rev-parse", "HEAD").strip()
    dirty = bool(git(root, "status", "--porcelain", "--untracked-files=no").strip())
    for k, v in [("commit", head), ("dirty", str(int(dirty))), ("built_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
                 ("facts_seconds", f"{t1 - t0:.1f}"), ("tables_seconds", f"{time.time() - t1:.1f}"), ("hash", "vivhash1")]:
        con.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (k, v))
    con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    free, pages = con.execute("PRAGMA freelist_count").fetchone()[0], con.execute("PRAGMA page_count").fetchone()[0]
    if free > pages // 4:
        con.execute("VACUUM")
    con.close()
    print(f"vivdb: built {db} at {head[:8]}{' (dirty)' if dirty else ''} in {time.time() - t0:.1f}s "
          f"({len(todo)} files read, {os.path.getsize(db) / 1e6:.1f} MB)", file=sys.stderr)


def rebuild_derived(con, root, blobs):
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for t in DERIVED_TABLES:
        con.execute(f"DROP TABLE IF EXISTS {t}")
    con.executescript(DERIVED)
    ins = lambda table, row: con.execute(f"INSERT OR IGNORE INTO {table} VALUES ({','.join('?' * len(row))})", row)

    for path, blob, text in con.execute("SELECT path, blob, json FROM facts_cache ORDER BY path").fetchall():
        f = json.loads(unz(text))
        m = f.get("module") or module_of(path)
        ins("files", (path, blob, "viv", now))
        ip = path[:-4] + ".intent"
        ins("modules", (m, path, blob, ip if ip in blobs else None, f.get("parse_error"), f.get("hash_error")))
        for i in f.get("imports", []):
            ins("imports", (m, i))
        for u in f.get("used_modules", []):
            ins("uses", (m, norm_module(u, path)))
        for d in f.get("defs", []):
            ins("defs", (d["ast_hash"], "def", len(d["params"]), d.get("signature")))
            # a `fn(...)` closure is lowered to a def of its own (`viv__fnbody_0`);
            # its span is the closure's
            kind = "closure" if d["name"].startswith("viv__fnbody") else "def"
            ins("def_sites", (d["ast_hash"], m, d["name"], kind, d["line_start"], d["line_end"],
                              d["byte_start"], d["byte_end"], d.get("ir_hash"), d["closure_hash"]))
            for k, p in enumerate(d["params"]):
                ins("params", (m, d["name"], k, p["name"], p["type"]))
            add_calls(ins, m, d["name"], "def", d["calls"], path)
            for k, s in enumerate(d["steps"]):
                ins("steps", (m, d["name"], k, s["id"], s["text"], s["depth"], s["line_start"], s["line_end"]))
            for doc in d["docs"]:
                ins("docs", (m, d["name"], doc["label"], doc["text"], doc["line"]))
            it = d["intent"]
            ins("intent", (m, d["name"], "", d["ast_hash"], it["stamp"], it["state"], it["source"]))
        if "main" in f:
            mn = f["main"]
            if mn["statements"]:
                ins("defs", (mn["ast_hash"], "main", 0, None))
                ins("def_sites", (mn["ast_hash"], m, "<main>", "main", None, None, None, None, mn.get("ir_hash"), None))
                add_calls(ins, m, "<main>", "main", mn["calls"], path)
        for t in f.get("tests", []):
            tid = f"{m}:{t['name']}"
            ins("tests", (tid, m, t["name"], t["line_start"], t["byte_start"], t["byte_end"], t["ast_hash"]))
            add_calls(ins, m, tid, "test", t["calls"], path)
        for u in f.get("features", []):
            ins("feature_uses", (m, u["feature"], u["count"]))
        for h in f.get("handwritten", []):
            ins("handwritten", (m, h["pattern"], h["feature"], h["line"]))
        if ip in blobs:
            ins("files", (ip, blobs[ip], "intent", now))

    # a test covers every def its calls reach (static)
    con.execute("""
      INSERT OR IGNORE INTO test_covers
      WITH RECURSIVE reach(test_id, module, name) AS (
        SELECT caller, callee_module, callee FROM calls WHERE caller_kind = 'test' AND callee_module IS NOT NULL
        UNION
        SELECT r.test_id, c.callee_module, c.callee FROM reach r
        JOIN calls c ON c.caller_module = r.module AND c.caller = r.name AND c.caller_kind = 'def'
        WHERE c.callee_module IS NOT NULL)
      SELECT test_id, module, name, 'static' FROM reach""")

    ingest_bugs(con, ins, root, blobs)
    ingest_exports(ins, root, blobs)
    ingest_artifacts(ins, blobs)
    ingest_claims(ins, root, blobs)
    ingest_gate(ins, root)
    con.commit()


def add_calls(ins, module, caller, kind, calls, path):
    for c in calls:
        base = c["name"].rsplit(".", 1)[-1]
        t = c["target"]
        cm = module if t == "local" else (norm_module(t[len("module:"):], path) if t.startswith("module:") else None)
        ins("calls", (module, caller, kind, c["name"], base, cm, t, int(t == "builtin"), int(c.get("value", False)), c["line"]))


def ingest_bugs(con, ins, root, blobs):
    path = "bugs/ledger.jsonl"
    if path not in blobs:
        return
    ins("files", (path, blobs[path], "ledger", time.strftime("%Y-%m-%dT%H:%M:%S")))
    commits = {}
    for line in open(os.path.join(root, path)):
        line = line.strip()
        if not line:
            continue
        b = json.loads(line)
        ins("bugs", (b.get("id"), b.get("title"), b.get("status"), b.get("found"), b.get("fixed"), b.get("found_by"),
                     json.dumps(b.get("engines")), int(bool(b.get("disagreement"))), b.get("commit"), b.get("test"), b.get("root_cause")))
        if b.get("commit"):
            commits.setdefault(b["commit"], []).append(b["id"])
    # the files each fix commit touched: one `git show` for all of them
    if not commits:
        return
    out = subprocess.run(["git", "-C", root, "show", "--format=@@%H", "--name-only", *commits],
                         capture_output=True, text=True).stdout
    sha = None
    for line in out.splitlines():
        if line.startswith("@@"):
            sha = line[2:].strip()
            continue
        if not line.strip() or sha is None:
            continue
        key = next((c for c in commits if sha.startswith(c)), None)
        for bid in commits.get(key, []):
            ins("bug_files", (bid, line, bug_module(line)))


def bug_module(path):
    if path.endswith(".viv"):
        return module_of(path)
    m = re.match(r"(lang/viv/src|crates/knotkit/src)/(.+)\.(rs|c)$", path)
    if m:
        return "compiler:" + m.group(2)
    return None


def ingest_exports(ins, root, blobs):
    for path in sorted(p for p in blobs if p.endswith(".export.toml")):
        with open(os.path.join(root, path), "rb") as fh:
            t = tomllib.load(fh)
        name = t.get("name") or os.path.basename(path)[: -len(".export.toml")]
        files = list(t.get("files", []))
        ins("exports", (name, path, t.get("about"), json.dumps(files)))
        for f in files:
            ins("export_files", (name, f, module_of(f), "file"))
        for case, d in (t.get("decks") or {}).items():
            for role in ("deck", "checks"):
                if isinstance(d, dict) and d.get(role):
                    ins("export_files", (name, d[role], module_of(d[role]), role))
        # a kernels export names the def behind each case: `viv = "file.viv:def"`
        for case, k in (t.get("kernels") or {}).items():
            if isinstance(k, dict) and ":" in k.get("viv", ""):
                f, d = k["viv"].rsplit(":", 1)
                ins("export_kernels", (name, case, module_of(f), d))
        hp = path[: -len(".export.toml")] + ".exact_hashes.json"
        if hp in blobs:
            for case, h in json.load(open(os.path.join(root, hp))).items():
                ins("export_hashes", (name, case, h if isinstance(h, str) else json.dumps(h)))


def ingest_artifacts(ins, blobs):
    """Generated files beside the programs that make them: `x.gen.json`
    beside `x.viv`; `d/wasm/x.wasm` for `d/x.viv` and the `d/x_*.viv` that
    drive it (dft/build-wasm.sh compiles `si_it_check.viv` over `si.viv`).
    The wasm manifests do not name their sources, so this is a naming rule."""
    by_dir = {}
    for q in blobs:
        if q.endswith(".viv"):
            by_dir.setdefault(os.path.dirname(q), []).append(os.path.basename(q))
    for p in blobs:
        d, base = os.path.split(p)
        if base.endswith(".gen.json"):
            cand = os.path.join(d, base[: -len(".gen.json")] + ".viv")
            if cand in blobs:
                ins("artifacts", (p, module_of(cand), "gen.json"))
        elif base.endswith(".wasm") and os.path.basename(d) == "wasm":
            stem, src = base[:-5], os.path.dirname(d)
            for f in by_dir.get(src, []):
                if f == stem + ".viv" or f.startswith(stem + "_"):
                    ins("artifacts", (p, module_of(os.path.join(src, f)), "wasm"))


def ingest_claims(ins, root, blobs):
    path = "www/src/data/claims.json"
    if path not in blobs:
        return
    try:
        d = json.load(open(os.path.join(root, path)))
    except (OSError, ValueError):
        return
    mod = {}
    for p in d.get("posts", []):
        m = "posts/" + module_of(p["file"])
        mod[p["slug"]] = m
        ins("posts", (p["slug"], m, p.get("title"), p.get("kind"), p.get("lane"), p.get("author")))
    for n in d.get("numbers", []):
        link = n.get("link") or {}
        ins("claims", (n["id"], n["post"], mod.get(n["post"]), n.get("line"), n.get("raw"), n.get("status"),
                       link.get("src"), link.get("name") or link.get("what")))


def ingest_gate(ins, root):
    """target/gate/history.jsonl of this worktree, and of the main checkout
    (where the gate's merge runs write it); read only."""
    common = git(root, "rev-parse", "--path-format=absolute", "--git-common-dir").strip()
    seen = set()
    gid = 0
    for base in [root, os.path.dirname(common)]:
        h = os.path.join(base, "target", "gate", "history.jsonl")
        if not os.path.exists(h) or os.path.realpath(h) in seen:
            continue
        seen.add(os.path.realpath(h))
        for line in open(h):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            gid += 1
            ins("gate_runs", (gid, r.get("time"), r.get("tier"), r.get("verdict"), r.get("seconds"), r.get("branch"),
                              r.get("sha"), int(bool(r.get("dirty"))), r.get("source")))
            for k, v in (r.get("checks") or {}).items():
                ins("gate_checks", (gid, k, v if isinstance(v, str) else json.dumps(v)))


# ---------------------------------------------------------------------------
# query and show
# ---------------------------------------------------------------------------

def connect_ro(db):
    if not os.path.exists(db):
        sys.exit(f"vivdb: no database at {db}; run `vivdb.py build` first")
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def saved_queries():
    qd = os.path.join(HERE, "queries")
    return {f[:-4]: os.path.join(qd, f) for f in sorted(os.listdir(qd)) if f.endswith(".sql")}


def run_query(con, sql, params):
    names = set(re.findall(r"(?<!:):([A-Za-z_]\w*)", sql))
    missing = names - set(params)
    if missing:
        sys.exit(f"vivdb: the query needs --param {' '.join(sorted(n + '=...' for n in missing))}")
    cur = con.execute(sql, {k: params[k] for k in names})
    cols = [c[0] for c in cur.description or []]
    return cols, cur.fetchall()


def print_table(cols, rows, as_json):
    if as_json:
        print(json.dumps([dict(zip(cols, r)) for r in rows], indent=1))
        return
    if not cols:
        return
    cells = [[("" if v is None else str(v)) for v in r] for r in rows]
    width = [min(60, max([len(c)] + [len(r[i]) for r in cells])) for i, c in enumerate(cols)]
    line = lambda vals: "  ".join(v[:60].ljust(width[i]) for i, v in enumerate(vals)).rstrip()
    print(line(cols))
    print(line(["-" * w for w in width]))
    for r in cells:
        print(line(r))
    print(f"({len(rows)} row{'s' if len(rows) != 1 else ''})")


def stale_note(con, root):
    meta = dict(con.execute("SELECT key, value FROM meta"))
    head = git(root, "rev-parse", "HEAD").strip()
    if meta.get("commit") != head:
        return f"note: the database was built at {meta.get('commit', '?')[:8]}, HEAD is {head[:8]}; rebuild for current answers"
    return None


def query(args):
    root = repo_root()
    con = connect_ro(args.db or default_db(root))
    params = dict(p.split("=", 1) for p in args.param)
    if args.saved:
        qs = saved_queries()
        if args.saved not in qs:
            sys.exit(f"vivdb: no saved query `{args.saved}`; saved: {', '.join(qs)}")
        sql = open(qs[args.saved]).read()
    elif args.sql:
        sql = args.sql
    else:
        sys.exit("vivdb: query takes SQL or --saved NAME")
    cols, rows = run_query(con, sql, params)
    note = stale_note(con, root)
    if note:
        print(note, file=sys.stderr)
    print_table(cols, rows, args.json)


def find_sites(con, ref):
    """`module.name`, `module:name` or a bare name."""
    for sep in (":", "."):
        if sep in ref:
            m, n = ref.rsplit(sep, 1)
            rows = con.execute("SELECT module, name, ast_hash, byte_start, byte_end, line_start FROM def_sites WHERE module = ? AND name = ?", (m, n)).fetchall()
            if rows:
                return rows
    return con.execute("SELECT module, name, ast_hash, byte_start, byte_end, line_start FROM def_sites WHERE name = ? ORDER BY module", (ref,)).fetchall()


def show(args):
    root = repo_root()
    con = connect_ro(args.db or default_db(root))
    sites = find_sites(con, args.ref)
    if not sites:
        sys.exit(f"vivdb: no def `{args.ref}`")
    if len(sites) > 1 and not args.all:
        print(f"`{args.ref}` is defined in {len(sites)} modules; showing the first (--all for every one): "
              + ", ".join(s[0] for s in sites), file=sys.stderr)
        sites = sites[:1]
    for module, name, h, b0, b1, l0 in sites:
        if args.as_ == "code":
            row = con.execute("SELECT b.bytes FROM modules m JOIN blobs b ON b.blob = m.blob WHERE m.module = ?", (module,)).fetchone()
            print(f"# {module}.viv:{l0}  ast {h[:12]}")
            sys.stdout.write(zlib.decompress(row[0])[b0:b1].decode() + "\n")
        elif args.as_ == "graph":
            print(f"{module}.{name}")
            callers = con.execute("SELECT DISTINCT caller_module, caller, caller_kind FROM calls WHERE callee_module = ? AND callee = ? ORDER BY 1, 2", (module, name)).fetchall()
            for cm, c, k in callers:
                print(f"  <- {c if k == 'test' else cm + '.' + c}" + ("" if k == "def" else f"  ({k})"))
            callees = con.execute("SELECT callee_name, target, MIN(line), value FROM calls WHERE caller_module = ? AND caller = ? GROUP BY callee_name, target ORDER BY 3", (module, name)).fetchall()
            for n, t, line, value in callees:
                print(f"  -> {n}  [{t}{', as a value' if value else ''}]  line {line}")
            for step, heading, depth, a, b in con.execute("SELECT step, heading, depth, line_start, line_end FROM steps WHERE module = ? AND def_name = ? ORDER BY ord", (module, name)):
                print(f"  {'  ' * depth}step {step or '-'}: {heading}  (lines {a}-{b})")
        else:
            st = con.execute("SELECT state, stamp, source FROM intent WHERE module = ? AND def_name = ? AND step_name = ''", (module, name)).fetchone()
            ir = con.execute("SELECT ir_hash FROM def_sites WHERE module = ? AND name = ?", (module, name)).fetchone()
            print(f"{module}.{name}: intent {st[0] if st else 'missing'}"
                  + (f", stamped @{st[1]}" if st and st[1] else "") + (f" ({st[2]})" if st and st[2] != 'none' else "")
                  + (f", body {ir[0]}" if ir and ir[0] else ""))
            for label, text, line in con.execute("SELECT label, text, line FROM docs WHERE module = ? AND def_name = ? ORDER BY line", (module, name)):
                print(f"  {label}: {text}")
            for step, heading, depth in con.execute("SELECT step, heading, depth FROM steps WHERE module = ? AND def_name = ? ORDER BY ord", (module, name)):
                print(f"  {'  ' * depth}{step or '-'}: {heading}")


def list_queries(_args):
    for name, path in saved_queries().items():
        first = next((l[3:].strip() for l in open(path) if l.startswith("-- ")), "")
        print(f"{name:28} {first}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vivdb.py", description=__doc__.split("\n")[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", help="the database file (default target/vivdb/viv.sqlite)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", parents=[common])
    b.add_argument("--viv", help="the compiler (default target/release/viv)")
    b.add_argument("--jobs", type=int, default=4)
    b.add_argument("--paths", nargs="*", help="only these files or directories")
    b.set_defaults(func=build)
    q = sub.add_parser("query", parents=[common])
    q.add_argument("sql", nargs="?")
    q.add_argument("--saved")
    q.add_argument("--json", action="store_true")
    q.add_argument("--param", action="append", default=[], help="k=v, bound to :k")
    q.set_defaults(func=query)
    s = sub.add_parser("show", parents=[common])
    s.add_argument("ref")
    s.add_argument("--as", dest="as_", choices=["code", "graph", "intent"], default="code")
    s.add_argument("--all", action="store_true")
    s.set_defaults(func=show)
    sub.add_parser("queries").set_defaults(func=list_queries)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
