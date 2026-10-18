-- SPDX-License-Identifier: MIT OR Apache-2.0
-- SPDX-FileCopyrightText: 2026 The Viv authors
-- Hand-written patterns that a feature replaces, with how often that feature is used (design 5.5).
SELECT h.pattern, h.feature, COUNT(*) AS handwritten, COUNT(DISTINCT h.module) AS modules
FROM handwritten h
GROUP BY h.pattern, h.feature
ORDER BY handwritten DESC;
