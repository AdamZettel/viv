-- SPDX-License-Identifier: MIT OR Apache-2.0
-- SPDX-FileCopyrightText: 2026 The Viv authors
-- Bugs per module, from the files each ledger entry's fix commit touched (design 5.9). Compiler sources show as compiler:<file>.
SELECT bf.module, b.status, COUNT(DISTINCT b.id) AS n, GROUP_CONCAT(DISTINCT b.id) AS ids
FROM bugs b JOIN bug_files bf ON bf.bug_id = b.id
WHERE bf.module IS NOT NULL
GROUP BY bf.module, b.status
ORDER BY n DESC, bf.module;
