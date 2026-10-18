-- SPDX-License-Identifier: MIT OR Apache-2.0
-- SPDX-FileCopyrightText: 2026 The Viv authors
-- Defs whose intent is missing, stale or unstamped (design 5.2). Names starting with _ are private helpers and left out.
SELECT s.module, s.name, i.state, i.stamp, i.source, s.ir_hash AS body, s.line_start
FROM def_sites s
JOIN intent i ON i.module = s.module AND i.def_name = s.name AND i.step_name = ''
WHERE s.kind = 'def'
  AND s.name NOT LIKE '\_%' ESCAPE '\'
  AND i.state <> 'current'
ORDER BY i.state, s.module, s.line_start;
