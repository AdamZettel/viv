-- SPDX-License-Identifier: MIT OR Apache-2.0
-- SPDX-FileCopyrightText: 2026 The Viv authors
-- Defs nothing calls or names: no call or value reference from any def, top level or test (design 5.4). Defs in an export's files are left out, since the export's users call them.
SELECT s.module, s.name, s.line_start
FROM def_sites s
WHERE s.kind = 'def'
  AND NOT EXISTS (SELECT 1 FROM calls c WHERE c.callee_module = s.module AND c.callee = s.name)
  AND NOT EXISTS (SELECT 1 FROM export_files ef WHERE ef.module = s.module AND ef.role = 'file')
  AND s.module NOT IN (SELECT module FROM modules WHERE parse_error IS NOT NULL)
ORDER BY s.module, s.line_start;
