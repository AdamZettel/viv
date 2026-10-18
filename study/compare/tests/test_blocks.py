# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ imports
import math

# @@ the code under test
def median(xs):
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])

def normalize(xs):
    total = sum(xs)
    return [x / total for x in xs]

# @@ one test per behavior
def test_median_of_an_odd_count():
    assert median([3.0, 1.0, 2.0]) == 2.0

def test_median_of_an_even_count():
    assert median([4.0, 1.0, 3.0, 2.0]) == 2.5

def test_normalized_weights_sum_to_one():
    w = normalize([0.1, 0.2, 0.3])
    assert math.isclose(sum(w), 1.0, abs_tol=1e-15)
    assert all(x > 0 for x in w)

def test_a_list_may_end_with_a_comma():
    xs = [
        1.0,
        2.0,
        3.0,
    ]
    assert len(xs) == 3

# @@ run them and report
TESTS = [test_median_of_an_odd_count, test_median_of_an_even_count,
         test_normalized_weights_sum_to_one, test_a_list_may_end_with_a_comma]
for t in TESTS:
    name = t.__name__.removeprefix("test_").replace("_", " ")
    try:
        t()
        print("PASS", name)
    except AssertionError:
        print("FAIL", name)
