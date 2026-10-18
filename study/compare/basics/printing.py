# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ print text and values
name = "Ada"
count = 3
print("hello,", name)
print("count:", count)
print("a", "b", "c", sep=", ")

# @@ the full digits of a float
x = 2.0 / 3.0
print(x)
print(f"x = {x}")
print(f"{x=}")

# @@ six significant digits
print(f"{x:g}")
print(f"{1e-7:g} {123456789.0:g}")

# @@ a format per value
print(f"x = {x:.3f}")
print(f"x = {x:.3e}")
print(f"[{x:10.4f}]")
print(f"count = {count:d}, padded = [{count:4d}]")

# @@ strings in quotes
word = "it's"
print(repr(word))
print(f"{word!r} has {len(word)} characters")

# @@ build a line piece by piece
line = ""
for i in range(count):
    line += f"{i:d} "
print(line.rstrip())

# @@ checks
assert f"{x:.2f}" == "0.67"
assert str(0.5) == "0.5"
