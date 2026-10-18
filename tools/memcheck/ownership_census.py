#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""A prototype census of aliasing in Viv programs.

    tools/memcheck/ownership_census.py OUT.tsv FILE.viv ...

Each program is lowered with `viv --emit ir` (the typed IR, with imports
modules expanded), and each def and the top level is walked in statement
order. The census reports, per program, how many sites of each kind it has:

  alias          `b = a` (or `b = r.f`) for a handle type (array, map, record
                 holding arrays or maps): two names on one storage.
  alias_mut      an alias after which one of the two names is written and the
                 other is read again: the program can see the sharing. Under
                 value semantics (section 4) its output would change.
  field_alias    a variable handed to a record literal as a field whose
                 elements are written through the variable afterwards: the
                 record and the variable share the storage.
  arg_overlap    a call passing one array, map or record (or a record and
                 one of its fields; two different fields do not overlap) to
                 two parameters of which the
                 callee writes at least one (or a builtin that writes one).
  global_write   a def that writes a top-level array or map (element write,
                 or passing it to something that writes it).
  global_conflict  a call that passes a top-level variable to
                 a callee that also writes that same variable as a global.
  returns_param  a def that returns one of its handle parameters, or a field
                 of one, without copying.
  par_nondisjoint  a `parallel for` that writes an array element whose
                 first index is not the loop variable, or writes through a def.
  inout_params   parameters a def writes (informational: these become
                 in-place parameters with no annotation).

It is an approximation, stated so the numbers can be read: it is
flow-insensitive inside a loop (a use anywhere in an enclosing loop counts
as "later"), a registry call (`apply`) is assumed to write nothing, it
follows names, not values (a handle that travels through a
def's return is not tracked), and builtins that write their arguments are
the list the compiler's graph pass uses (lang/viv/src/graph.rs). The top
level's variables are globals; a def's writes to them are found through the
names the def does not bind.
"""
import os
import re
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIV = os.environ.get("VIV", os.path.join(ROOT, "target/release/viv"))

# runtime callees that write an argument: C name -> first argument written
# (every later argument that is a variable counts, as in graph.rs)
WRITES = {
    "viv_eig_sym": 1, "viv_eig_herm": 1, "viv_eig_general": 1, "viv_eig_near": 1,
    "viv_spectrum_sym": 1, "viv_cholesky": 1, "viv_lin_solve": 2, "viv_lin_solve_multi": 2,
    "viv_put": 0, "viv_setslab": 0, "viv_setslice": 0, "viv_setslice_n": 0, "viv_setslice_c": 0,
    "viv_sort": 0, "viv_fill": 0, "viv_cfill": 0, "viv_fft3": 0, "viv_ifft3": 0,
    "viv_map_clear": 0,
}
WRITE_PREFIX = ("viv_map_set", "viv_map_remove")
LINE = re.compile(r"^( *)([0-9a-f]{8})  (\w+)(.*)$")


class Node:
    __slots__ = ("kind", "attr", "kids", "h")

    def __init__(self, kind, attr, h):
        self.kind, self.attr, self.kids, self.h = kind, attr, [], h


def parse_ir(text):
    roots, stack = [], []
    for line in text.split("\n"):
        m = LINE.match(line)
        if not m:
            continue
        depth = len(m.group(1)) // 2
        n = Node(m.group(3), m.group(4), m.group(2))
        del stack[depth:]
        if stack:
            stack[-1].kids.append(n)
        else:
            roots.append(n)
        stack.append(n)
    return roots


def attr_str(n, key):
    m = re.search(key + r': "([^"]*)"', n.attr)
    return m.group(1) if m else None


def typed_list(attr, key):
    """`key: [("a", T), ...]` from a node's attributes, T possibly nested."""
    at = attr.find(key + ": [")
    if at < 0:
        return {}
    i = at + len(key) + 3
    out, depth, start = {}, 0, None
    while i < len(attr):
        ch = attr[i]
        if ch in "([{":
            if depth == 0 and ch == "(":
                start = i
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0 and ch == ")" and start is not None:
                item = attr[start + 1:i]
                m = re.match(r'"([^"]+)", (.*)$', item)
                if m:
                    out[m.group(1)] = m.group(2)
                start = None
            if depth < 0:
                break
        i += 1
    return out


def is_handle(ty, records):
    if ty is None:
        return False
    if ty.startswith("Arr(") and ty != "Arr(Str)" and not ty.startswith("Arr(Func"):
        return True
    if ty.startswith("Map("):
        return True
    m = re.match(r'Record\("([^"]+)"\)', ty)
    if m:
        return records.get(m.group(1), False)
    return False


def root_var(n):
    """The variable an lvalue-ish expression starts from: a, a.f, a.f.g."""
    while n.kind == "Field" and n.kids:
        n = n.kids[0]
    if n.kind == "Var":
        return attr_str(n, "name")
    return None


def path_of(n):
    """`a`, `a.f`, `a.f.g` for a variable or a chain of field reads, else None."""
    fields = []
    while n.kind == "Field" and n.kids:
        fields.append(attr_str(n, "field"))
        n = n.kids[0]
    if n.kind != "Var":
        return None
    return ".".join([attr_str(n, "name")] + fields[::-1])


def walk(n, f):
    f(n)
    for k in n.kids:
        walk(k, f)


class Census:
    def __init__(self, roots):
        self.funcs, self.records, self.globals = {}, {}, {}
        self.main = None
        rec_fields = {}
        for r in roots:
            if r.kind == "RecordDef":
                name = attr_str(r, "name")
                rec_fields[name] = r.attr
            elif r.kind == "Func":
                self.funcs[attr_str(r, "name")] = r
            elif r.kind == "Main":
                self.main = r
            elif r.kind == "Global":
                m = re.search(r'name: "([^"]+)", ty: (.*) \}$', r.attr)
                if m:
                    self.globals[m.group(1)] = m.group(2)
        # a record type is a handle when a field (transitively) is one
        changed = True
        while changed:
            changed = False
            for name, attr in rec_fields.items():
                h = bool(re.search(r"Arr\((Num|Cplx|Func)|Map\(", attr)) or any(
                    self.records.get(o) for o in re.findall(r'Record\("([^"]+)"\)', attr))
                if h and not self.records.get(name):
                    self.records[name] = True
                    changed = True
        self.params, self.locals = {}, {}
        for name, f in self.funcs.items():
            self.params[name] = list(typed_list(f.attr, "params").items())
            self.locals[name] = typed_list(f.attr, "locals")
        # which parameter positions each def writes (fixpoint through calls)
        self.writes_param = {n: set() for n in self.funcs}
        self.writes_global = {n: set() for n in self.funcs}
        changed = True
        while changed:
            changed = False
            for name, f in self.funcs.items():
                pnames = [p for p, _ in self.params[name]]
                bound = set(pnames) | set(self.locals[name])
                for v in self.written_vars(f):
                    if v in pnames and pnames.index(v) not in self.writes_param[name]:
                        self.writes_param[name].add(pnames.index(v))
                        changed = True
                    if v not in bound and v in self.globals and v not in self.writes_global[name]:
                        self.writes_global[name].add(v)
                        changed = True
                for callee in self.callees(f):
                    for g in self.writes_global.get(callee, ()):
                        if g not in bound and g not in self.writes_global[name]:
                            self.writes_global[name].add(g)
                            changed = True

    def callees(self, f):
        out = []
        walk(f, lambda n: out.append(attr_str(n, "callee")) if n.kind == "Call" else None)
        return [c for c in out if c in self.funcs]

    def call_writes(self, n):
        """The argument positions a Call node writes."""
        callee = attr_str(n, "callee")
        if callee in self.funcs:
            return self.writes_param[callee]
        if callee in WRITES:
            return set(range(WRITES[callee], len(n.kids)))
        if callee and callee.startswith(WRITE_PREFIX):
            return {0}
        return set()

    def writes_of(self, n):
        """Variables a single node writes through a handle (not rebinding)."""
        out = []
        if n.kind in ("SetIndex", "CSetIndex", "Scatter", "Fft") and n.kids:
            v = root_var(n.kids[0])
            if v:
                out.append(v)
        elif n.kind == "Call":
            for j in self.call_writes(n):
                if j < len(n.kids):
                    v = root_var(n.kids[j])
                    if v:
                        out.append(v)
        return out

    def written_vars(self, f):
        out = set()
        walk(f, lambda n: out.update(self.writes_of(n)))
        return out


def events(body, census, out=None):
    """Linearize a def body: (index, kind, name, node, loop_ids)."""
    if out is None:
        out = []

    def rec(n, loop_stack):
        if n.kind in ("For", "While", "ParallelFor"):
            loop_stack = loop_stack + [id(n)]
        if n.kind == "Var":
            out.append(("read", attr_str(n, "name"), n, tuple(loop_stack)))
        for w in census.writes_of(n):
            out.append(("write", w, n, tuple(loop_stack)))
        if n.kind == "Assign":
            # the value is evaluated before the name is rebound
            for k in n.kids:
                rec(k, loop_stack)
            out.append(("bind", attr_str(n, "name"), n, tuple(loop_stack)))
            return
        for k in n.kids:
            rec(k, loop_stack)

    for k in body.kids:
        rec(k, [])
    return out


def later(ev, i, pred):
    """Events after position i, plus every event in a loop enclosing i."""
    loops = set(ev[i][3])
    for j, e in enumerate(ev):
        if j > i or (loops and loops & set(e[3])):
            if j != i and pred(e):
                yield j, e


def analyze(text):
    roots = parse_ir(text)
    c = Census(roots)
    sites = defaultdict(list)  # kind -> [(def, detail)]
    bodies = [(name, f, dict(c.params[name]), c.locals[name]) for name, f in c.funcs.items()]
    if c.main is not None:
        bodies.append(("<top>", c.main, {}, {}))
    for name, f, params, locs in bodies:
        types = dict(c.globals)
        types.update(locs)
        types.update(params)
        ev = events(f, c)
        for i, (kind, var, node, _) in enumerate(ev):
            if kind != "bind" or not node.kids:
                continue
            src = root_var(node.kids[0]) if node.kids[0].kind in ("Var", "Field") else None
            if not src or src == var or not is_handle(types.get(var), c.records):
                continue
            sites["alias"].append((name, f"{var} = {src}"))
            seen = False
            for j, e in later(ev, i, lambda e: e[0] == "write" and e[1] in (var, src)):
                # a name rebound between the alias and the write no longer shares
                if j > i and any(ev[t][0] == "bind" and ev[t][1] == e[1] for t in range(i + 1, j)):
                    continue
                other = src if e[1] == var else var
                if any(k > j or (set(e[3]) & set(e2[3])) for k, e2 in later(ev, i, lambda e2: e2[0] == "read" and e2[1] == other)):
                    seen = True
                    break
            if seen:
                sites["alias_mut"].append((name, f"{var} = {src}"))
        # record literal fields that capture a handle variable written elsewhere
        pos = {id(e[2]): i for i, e in enumerate(ev) if e[0] == "read"}

        def rec_lits(n):
            if n.kind == "RecordLit":
                for k in n.kids:
                    if k.kind != "Var" or id(k) not in pos:
                        continue
                    v = attr_str(k, "name")
                    if not is_handle(types.get(v), c.records):
                        continue
                    i = pos[id(k)]
                    for j, e in later(ev, i, lambda e: e[0] == "write" and e[1] == v):
                        if j > i and any(ev[t][0] == "bind" and ev[t][1] == v for t in range(i + 1, j)):
                            continue
                        sites["field_alias"].append((name, f"{attr_str(n, 'ty_name')} holds {v}, and {v} is written after"))
                        break
            for k in n.kids:
                rec_lits(k)
        rec_lits(f)

        def calls(n):
            if n.kind == "Call":
                wr = c.call_writes(n)
                if wr:
                    paths = [path_of(k) for k in n.kids]
                    paths = [p if p and is_handle(types.get(p.split(".")[0]), c.records) else None for p in paths]
                    for j in wr:
                        if j < len(paths) and paths[j]:
                            pj = paths[j]
                            if any(k != j and q and (q == pj or q.startswith(pj + ".") or pj.startswith(q + "."))
                                   for k, q in enumerate(paths)):
                                sites["arg_overlap"].append((name, f"{attr_str(n, 'callee')}(... {pj} ... overlapping ...)"))
                                break
                callee = attr_str(n, "callee")
                if callee in c.funcs:
                    for k in n.kids:
                        v = root_var(k) if k.kind in ("Var", "Field") else None
                        if v and v in c.globals and v not in params and v not in locs and v in c.writes_global[callee]:
                            sites["global_conflict"].append((name, f"{callee}({v}) while {callee} writes the global {v}"))
            if n.kind == "ParallelFor":
                var = attr_str(n, "var")
                body = n.kids[1] if len(n.kids) > 1 else n

                def pw(m):
                    if m.kind in ("SetIndex", "CSetIndex") and len(m.kids) >= 2:
                        base = root_var(m.kids[0])
                        idx = m.kids[1]
                        if not (idx.kind == "Var" and attr_str(idx, "name") == var):
                            sites["par_nondisjoint"].append((name, f"parallel for {var}: {base}[...] written at an index that is not {var}"))
                    if m.kind == "Call" and attr_str(m, "callee") in c.funcs and c.call_writes(m):
                        sites["par_nondisjoint"].append((name, f"parallel for {var}: calls {attr_str(m, 'callee')}, which writes an argument"))
                    for k in m.kids:
                        pw(k)
                pw(body)
            for k in n.kids:
                calls(k)
        calls(f)
        if name != "<top>":
            if c.writes_global[name]:
                sites["global_write"].append((name, ", ".join(sorted(c.writes_global[name]))))
            for j in sorted(c.writes_param[name]):
                sites["inout_params"].append((name, c.params[name][j][0]))

            def rets(n):
                if n.kind == "Return" and n.kids:
                    v = root_var(n.kids[0]) if n.kids[0].kind in ("Var", "Field") else None
                    if v in params and is_handle(params[v], c.records):
                        sites["returns_param"].append((name, f"returns {v}"))
                for k in n.kids:
                    rets(k)
            rets(f)
    return sites, c


KINDS = ["alias", "alias_mut", "field_alias", "arg_overlap", "global_write", "global_conflict",
         "returns_param", "par_nondisjoint", "inout_params"]


def main():
    out = sys.argv[1]
    per_def = defaultdict(set)  # kind -> {(def, detail)} over the corpus, each def once
    rows = []
    for path in sys.argv[2:]:
        r = subprocess.run([VIV, "--emit", "ir", path], capture_output=True, text=True)
        if r.returncode != 0:
            rows.append((path, None, None))
            continue
        sites, c = analyze(r.stdout)
        rows.append((path, sites, c))
        for k, v in sites.items():
            for s in v:
                per_def[k].add(s)
    with open(out, "w") as f:
        f.write("program\t" + "\t".join(KINDS) + "\n")
        for path, sites, _ in rows:
            rel = os.path.relpath(path, ROOT)
            if sites is None:
                f.write(rel + "\t" + "\t".join("-" for _ in KINDS) + "\n")
                continue
            f.write(rel + "\t" + "\t".join(str(len(sites.get(k, []))) for k in KINDS) + "\n")
    ok = [r for r in rows if r[1] is not None]
    print(f"programs lowered: {len(ok)} of {len(rows)}")
    print(f"{'kind':18} {'programs':>8} {'distinct sites':>15}")
    for k in KINDS:
        n = sum(1 for _, s, _ in ok if s.get(k))
        print(f"{k:18} {n:8d} {len(per_def[k]):15d}")
    with open(os.path.splitext(out)[0] + "_sites.txt", "w") as f:
        for k in KINDS:
            if k == "inout_params":
                continue
            f.write(f"== {k} ({len(per_def[k])} distinct sites)\n")
            for d, detail in sorted(per_def[k]):
                f.write(f"  {d}: {detail}\n")


if __name__ == "__main__":
    main()
