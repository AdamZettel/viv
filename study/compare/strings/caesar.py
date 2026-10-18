# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ shift one letter
ALPHABET = 26

def shift_char(c, k):
    if "a" <= c <= "z":
        return chr((ord(c) - ord("a") + k) % ALPHABET + ord("a"))
    if "A" <= c <= "Z":
        return chr((ord(c) - ord("A") + k) % ALPHABET + ord("A"))
    return c

# @@ shift a message
def caesar(text, k):
    return "".join(shift_char(c, k) for c in text)

# @@ encode and decode
KEY = 3
message = "Veni, vidi, vici. Zebra!"
secret = caesar(message, KEY)
print(secret)
print(caesar(secret, -KEY))

# @@ break it by trying every key
for k in range(ALPHABET):
    guess = caesar(secret, -k)
    if "vidi" in guess:
        print("key", k, "gives", guess)

# @@ checks
assert caesar(caesar(message, 11), -11) == message
assert caesar("xyz", KEY) == "abc"
