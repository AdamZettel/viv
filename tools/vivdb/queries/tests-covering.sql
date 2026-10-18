-- SPDX-License-Identifier: MIT OR Apache-2.0
-- SPDX-FileCopyrightText: 2026 The Viv authors
-- The tests whose calls reach :def (module.name), statically (design 5.6). Run results join here once test_results exists (phase 4).
SELECT t.test_id, tc.how, t.line
FROM test_covers tc JOIN tests t ON t.test_id = tc.test_id
WHERE tc.module || '.' || tc.name = :def
ORDER BY t.module, t.line;
