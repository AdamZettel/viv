# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ the text
TEXT = "the cat sat on the mat and the dog sat on the log"

# @@ count each word
counts = {}
for w in TEXT.split():
    counts[w] = counts.get(w, 0) + 1
print(counts)

# @@ read the map
print(len(counts), "distinct words")
print("the:", counts["the"], " zebra:", counts.get("zebra", 0))
for w in ["dog", "zebra"]:
    print(w, "is a key" if w in counts else "is not a key")

# @@ the most frequent words
TOP = 3
ranked = sorted(counts, key=lambda w: (-counts[w], w))
for w in ranked[:TOP]:
    print(f"{w:>4} {counts[w]}")

# @@ remove a key
n = counts.pop("and")
print("removed and:", n, "left:", list(counts))

# @@ checks
assert counts["sat"] == 2
assert sum(counts.values()) == len(TEXT.split()) - 1
