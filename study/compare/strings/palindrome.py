# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ keep letters and digits, in lower case
def normalize(s):
    return "".join(c.lower() for c in s if c.isalnum())

# @@ compare with the reverse
def is_palindrome(s):
    t = normalize(s)
    return t == t[::-1]

# @@ compare from both ends
def is_palindrome_two_ends(s):
    t = normalize(s)
    i, j = 0, len(t) - 1
    while i < j:
        if t[i] != t[j]:
            return False
        i += 1
        j -= 1
    return True

# @@ try some phrases
PHRASES = [
    "A man, a plan, a canal: Panama",
    "Was it a car or a cat I saw?",
    "No lemon, no melon",
    "Hello, world",
    "Été",
]
for p in PHRASES:
    verdict = "yes" if is_palindrome(p) else "no"
    print(f"{verdict:3} {normalize(p)}")

# @@ checks
for p in PHRASES:
    assert is_palindrome(p) == is_palindrome_two_ends(p)
