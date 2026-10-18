# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ the text
LINES = [
    "It was the best of times, it was the worst of times,",
    "it was the age of wisdom, it was the age of foolishness,",
    "it was the epoch of belief, it was the epoch of incredulity.",
]
TEXT = "\n".join(LINES)

# @@ lines, words and characters
lines = TEXT.splitlines()
words = TEXT.split()
print(len(lines), "lines,", len(words), "words,", len(TEXT), "characters")

# @@ clean each word
def clean(word):
    return word.strip(".,;:!?").lower()

cleaned = [clean(w) for w in words]
print(cleaned[:6])

# @@ the longest word
longest = max(cleaned, key=len)
print("longest:", longest, len(longest))

# @@ words per line
for n, line in enumerate(lines, start=1):
    print(f"line {n}: {len(line.split())} words")

# @@ checks
assert cleaned.count("was") == 6
assert longest == "foolishness"
