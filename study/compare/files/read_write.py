# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ imports
from pathlib import Path

# @@ write a file
PATH = "compare_scores.csv"
rows = ["name,score", "ann,72", "bob,95", "cy,88"]
n = Path(PATH).write_text("\n".join(rows) + "\n")
print("wrote", n, "characters")

# @@ read it back line by line
with open(PATH) as f:
    lines = f.read().splitlines()
print(len(lines), "lines; header:", lines[0])

# @@ parse the rows
names = []
scores = []
for line in lines[1:]:
    name, score = line.split(",")
    names.append(name)
    scores.append(float(score))
best = scores.index(max(scores))
print(f"best: {names[best]} with {scores[best]:.0f}")

# @@ write a report
report = [f"{nm}: {'*' * int(s // 10)}" for nm, s in zip(names, scores)]
Path("compare_report.txt").write_text("\n".join(report) + "\n")
print(Path("compare_report.txt").read_text(), end="")

# @@ checks
assert Path(PATH).read_text().count("\n") == len(rows)
