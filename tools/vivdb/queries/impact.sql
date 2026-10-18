-- SPDX-License-Identifier: MIT OR Apache-2.0
-- SPDX-FileCopyrightText: 2026 The Viv authors
-- What changing :target moves (design 5.3). :target is a builtin name ('exp') or a def as module.name ('lib/hf/scf.rhf').
-- Rows: every transitive caller (def, top level, test); the programs that run one or import a module whose top level does; the exports, generated artifacts and posts' numbers of those programs.
WITH RECURSIVE
hit(module, name, kind, depth) AS (
  SELECT caller_module, caller, caller_kind, 1 FROM calls
   WHERE (builtin = 1 AND callee = :target)
      OR (builtin = 0 AND callee_module || '.' || callee = :target)
  UNION
  SELECT c.caller_module, c.caller, c.caller_kind, h.depth + 1
  FROM calls c JOIN hit h ON c.callee_module = h.module AND c.callee = h.name
  WHERE h.kind = 'def' AND h.depth < 64
),
callers AS (SELECT module, name, kind, MIN(depth) AS depth FROM hit GROUP BY module, name, kind),
-- a module is affected when its top level or a test reaches the target,
-- or (for a library that an export ships) when one of its defs does
-- importing a module runs its top level, so a program that imports one
-- whose top level is affected is affected too (`from dft.pwdft_it import ETOT`)
programs(module) AS (
  SELECT module FROM callers WHERE kind IN ('main', 'test')
  UNION
  SELECT u.module FROM uses u JOIN programs p ON u.used = p.module
  WHERE p.module IN (SELECT module FROM callers WHERE kind = 'main')
     OR p.module IN (SELECT module FROM uses)
),
touched AS (SELECT DISTINCT module FROM callers),
hit_exports AS (
  SELECT DISTINCT ef.export FROM export_files ef
  WHERE (ef.role = 'file' AND ef.module IN (SELECT module FROM touched))
     OR (ef.role <> 'file' AND ef.module IN (SELECT module FROM programs))
),
-- an export's decks and checks are compiled over its files (export.toml),
-- with no import in their source: they run whatever the files run
run AS (
  SELECT module FROM programs
  UNION
  SELECT module FROM export_files WHERE role <> 'file' AND export IN (SELECT export FROM hit_exports)
)
SELECT 'def' AS what, module || '.' || name AS item, depth AS detail
  FROM callers WHERE kind = 'def'
UNION ALL
SELECT 'program', module,
       (SELECT GROUP_CONCAT(kind, ',') FROM (SELECT DISTINCT kind FROM callers c WHERE c.module = p.module AND c.kind <> 'def'))
  FROM run p
UNION ALL
SELECT 'export', e.export || ':' || h.case_name, h.exact_hash
  FROM hit_exports e JOIN export_hashes h ON h.export = e.export
  -- a kernels export: only the cases whose def reaches the target
  WHERE NOT EXISTS (SELECT 1 FROM export_kernels k WHERE k.export = h.export AND k.case_name = h.case_name)
     OR EXISTS (SELECT 1 FROM export_kernels k JOIN callers c ON c.module = k.module AND c.name = k.def_name
                WHERE k.export = h.export AND k.case_name = h.case_name)
UNION ALL
SELECT 'artifact', a.path, MIN(a.kind)
  FROM artifacts a WHERE a.module IN (SELECT module FROM run) GROUP BY a.path
UNION ALL
SELECT 'post', p.slug, (SELECT COUNT(*) FROM claims c WHERE c.post = p.slug AND c.status IN ('checked', 'computed', 'mismatch')) || ' numbers from runs'
  FROM posts p WHERE p.module IN (SELECT module FROM run)
ORDER BY 1, 2;
