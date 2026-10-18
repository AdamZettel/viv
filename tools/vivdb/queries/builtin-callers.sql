-- SPDX-License-Identifier: MIT OR Apache-2.0
-- SPDX-FileCopyrightText: 2026 The Viv authors
-- The builtins the corpus calls, by how many call sites and modules.
SELECT callee AS builtin, COUNT(*) AS sites, COUNT(DISTINCT caller_module) AS modules
FROM calls WHERE builtin = 1 GROUP BY callee ORDER BY sites DESC;
