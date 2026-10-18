# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ a function that can fail
def parse_age(text):
    value = int(text)
    if value < 0:
        raise ValueError(f"age cannot be negative: {value}")
    return value

# @@ the caller handles the failure
for text in ["42", "-3", "abc", "7"]:
    try:
        age = parse_age(text)
        print(f"{text!r}: age {age}")
    except ValueError as e:
        print(f"{text!r}: refused ({e})")

# @@ division by zero
def ratio(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return float("nan")

print(ratio(1.0, 4.0), ratio(1.0, 0.0))

# @@ a missing key
stock = {"apple": 3, "pear": 0}
for item in ["apple", "plum"]:
    try:
        print(item, stock[item])
    except KeyError:
        print(item, "not stocked")

# @@ a check that should never fail
total = sum(stock.values())
if total != 3:
    raise RuntimeError(f"stock total is {total}, expected 3")
print("stock total checked")

# @@ checks
assert parse_age("42") == 42
