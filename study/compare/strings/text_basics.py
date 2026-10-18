# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ a string and its pieces
s = "  The quick brown fox  "
t = s.strip()
print(f"[{t}] has {len(t)} characters")
print(t.upper())
print(t.lower())
print(t[4:9], t[-3:], t[::-1])

# @@ split and join
words = t.split()
print(words)
print("-".join(words))
print("a,b,,c".split(","))

# @@ search and replace
print(t.find("quick"), t.find("slow"))
if t.startswith("The") and t.endswith("fox"):
    print("starts with The, ends with fox")
print(t.replace("quick", "slow"))
print(t.count("o"))

# @@ padding and alignment
for w in words:
    print(f"|{w:<8}|{w:>8}|{w:^8}|")

# @@ characters one at a time
vowels = 0
for c in t:
    if c.lower() in "aeiou":
        vowels += 1
print("vowels:", vowels)

# @@ checks
assert "fox" in t
assert t.split(" ")[1] == "quick"
