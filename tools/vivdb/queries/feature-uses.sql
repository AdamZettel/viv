-- SPDX-License-Identifier: MIT OR Apache-2.0
-- SPDX-FileCopyrightText: 2026 The Viv authors
-- Each catalog feature: total uses and how many modules use it, fewest first.
SELECT feature, SUM(count) AS uses, COUNT(*) AS modules
FROM feature_uses GROUP BY feature ORDER BY uses ASC, feature;
