# TheAlgorithms/Python in Viv

Source: https://github.com/TheAlgorithms/Python @ `b3233c9f49f705757ea1626891725af01473fa66`
(shallow clone at `viv-wt/src/thealgorithms-python`).

40 functions and classes (about 25 distinct algorithms) from 27 files, in seven Viv programs. Each has tests (the file's doctests
and `__main__` asserts). Each prints the values the reference prints, in the same format.

| Viv file | ported from | algorithms |
|---|---|---|
| `sorting.viv` | sorts/merge_sort.py, quick_sort.py, heap_sort.py, insertion_sort.py; searches/binary_search.py | merge sort, quick sort (random pivot, 3-way), heap sort, insertion sort, leftmost binary search, bisect_left, recursive binary search |
| `graphs.viv` | graphs/breadth_first_search.py, depth_first_search.py, dijkstra.py, kahns_algorithm_topo.py, minimum_spanning_tree_kruskal.py; data_structures/disjoint_set/disjoint_set.py | BFS, DFS, Dijkstra (with a binary heap), Kahn topological sort, union-find (rank + path compression), Kruskal MST |
| `dp.viv` | dynamic_programming/knapsack.py, longest_common_subsequence.py, edit_distance.py | 0/1 knapsack (bottom-up, memoized, solution reconstruction), LCS with the subsequence, edit distance (top-down and bottom-up) |
| `strings.viv` | strings/knuth_morris_pratt.py, palindrome.py, check_anagrams.py | KMP + failure array, five palindrome checks, anagram check |
| `maths.viv` | maths/prime_sieve_eratosthenes.py, greatest_common_divisor.py, binary_exponentiation.py | sieve, gcd (recursive, iterative), four binary exponentiations |
| `structures.viv` | data_structures/stacks/stack.py, queues/queue_by_list.py, heap/heap.py, linked_list/singly_linked_list.py, hashing/hash_map.py | stack, queue (ring buffer), max-heap class, singly linked list, open-addressing hash map with resize |
| `backtracking.viv` | backtracking/n_queens.py, all_permutations.py | n-queens, permutations |
| `seq.viv` | (shared) | `slice`, `copy`, `concat`, stable `argsort`: things Python has built in (`pymod`, `floordiv` and `same` were here until Viv gained `%`, `//` and `all(a == b)`) |

Other files: `ref.py` runs the original code and writes `ref/<name>.out`. `run.sh` runs
every port in the interpreter and as native C (and, with `VIV_WASM=...`, through the direct
wasm backend in node) and compares the outputs. `count_lines.py` counts lines. `probe/`
holds one small program per friction item, with `probe/<name>.out` showing what each engine
did with it.

```
/opt/homebrew/bin/python3.12 ref.py
VIV_WASM=$PWD/../../../www/public/viv/viv.wasm ./run.sh
```

## Sizes

These are the inputs from the reference's doctests and `__main__` blocks, not reduced: 13-item
sort input, 7-vertex graphs, 8-queens (all 92 solutions, plus the counts for n = 1..8), a
12-key hash map with two resizes, and so on. Every program takes under 0.1 s in the
interpreter (`backtracking` 0.08 s, the rest under 0.01 s).

## Agreement

`ref.py` imports each reference file unmodified (`load()` makes two syntax-only changes so
Python 3.12 can compile code written for 3.14: `except A, B:` gets parentheses and
annotations are made lazy). It calls the reference functions on the port's inputs and prints
each result in Viv's `show` format. `run.sh` compares the bytes against the interpreter's
output with the PASS lines removed.

| port | lines compared | result |
|---|---|---|
| sorting | 7 | identical |
| graphs | 21 | identical |
| dp | 13 | identical |
| strings | 22 | identical |
| maths | 12 | 11 identical, 1 differs (see F14) |
| structures | 77 | identical |
| backtracking | 39 | identical |

The tolerance is zero: every value is an integer, a boolean, a character, or 5.0625/2.0625,
and all of them match exactly. Quick sort picks a random pivot in both versions. The two
random streams differ, but the sorted output is the same.

The one line that differs:
`bench_mod_iterative` is `binary_exp_mod_iterative(1269380576, 374, 34)`, from the file's
`__main__` benchmark. The original squares `base` without reducing it mod the modulus. In
Python that is an exact big integer, and the answer is 8. In a double it overflows to
`inf` after six squarings, and the answer is `nan`. The port reproduces the algorithm
exactly and gets the wrong result, which is the finding. `binary_exp_iterative(1269380576, 374)`
is a 3,400-digit integer in Python and `inf` in Viv; `ref.py` prints `inf` for that line
to record that.

## Engines

| port | interpreter | native C (`cc -O2`) | direct wasm (node) |
|---|---|---|---|
| all seven | all 25 tests pass | byte-identical to the interpreter | byte-identical to the interpreter |

No engine failed to run any port. The ports avoid F1 by nesting every bounds guard; written
the Python way, those guards run in C and stop the interpreter.

## Line counts (code lines: no blanks, comments, docstrings; ported functions only)

`count_lines.py` counts the reference functions and classes the ports cover, by AST range
(printing-only methods such as `__repr__` and `print_graph` are left out), and the lines
inside Viv `def`s. It does not count tests, `__main__` blocks, or runs.

| port | Python | Viv |
|---|---|---|
| sorting | 102 | 172 |
| graphs | 110 | 264 |
| dp | 121 | 143 |
| strings | 64 | 193 |
| maths | 59 | 89 |
| structures | 267 | 352 |
| backtracking | 53 | 70 |
| seq.viv (shared helpers) | – | 79 |
| **total** | **776** | **1362** |

The Python count includes type annotations and input validation that raises exceptions.
The port leaves out the validation (F9), so the real difference is larger than 1.75×.
Most of the extra Viv lines are in three places. Strings (3×) come from helpers Python has
built in (`lower`, `strip`, `replace`, `[::-1]`, `find`). Graphs (2.4×) come from the heap
that `heapq` provides and the CSR construction that `dict[list]` provides. Filters are two
passes instead of one comprehension. DP and backtracking stay close to the Python.

## Per-algorithm friction

| algorithm | friction hit |
|---|---|
| merge sort | F3 (slices), F11 (nested `merge`), F1 (merge guard) |
| quick sort | F10 (comprehensions over a closure), F2, F3 (`[*a, *b, *c]`) |
| heap sort | F1 (2 guards), F13 (descending ranges), F12 |
| insertion sort | F1 (`while j >= 0 and a[j] > key`), F13 |
| binary search ×3 | F11 (nested recursive helper), F9 (unsorted-input ValueError dropped) |
| BFS | F5 (dict of lists), F6 (adjacency → CSR), F16 (visited set → mask) |
| DFS | F5 (letter keys → codes), F6, F13 (`reversed(graph[v])` → descending while) |
| Dijkstra | F17 (heapq, 60 lines), F8 (heappop's return), F5, F6, F1 |
| Kahn topo | F5 (dict with arbitrary int keys → `index_of` linear search), F9 (`None` → empty array), F2 |
| union-find | F7 (Node objects → parallel arrays) |
| Kruskal | F17 (`sorted(key=)` → argsort), F11 (nested `find_parent`), F2 (tree trimmed row by row) |
| knapsack ×3 | F8 (`(value, table)`, `(value, set)`), F5 (set → mask), F2 |
| LCS | F8 (`(len, seq)`), F4 (string prepend), F20 (empty strings), F15 |
| edit distance ×2 | F7 (class with `self.dp`), F20; F4 zero-width matrix works (win) |
| KMP | F4 (list of probe strings → separator-joined), F2 (`failure.append`), F17 (`str.find`) |
| palindrome ×5 | F4 (`[::-1]`, `lower`, `isalnum`, dict of test strings), F3 (`s[1:-1]`), F20 |
| anagram | F4 (`lower().strip().replace()`), F5 (`defaultdict(int)` → 128 counts) |
| sieve | F2 (comprehension → count then fill), F16 |
| gcd ×2 | F14 (`%` is fmod; `pymod` keeps Python's semantics) |
| binary exp ×4 | F14 (bigints, `//`, `%`, `&`, `>>=`), F15 (`3.45227e+13`) |
| stack | F7, F9 (StackOverflowError / StackUnderflowError) |
| queue | F7, F17 (`list.pop(0)` → ring buffer) |
| max-heap class | F7, F1 (2 guards around `None` children), F14 (`(idx - 1) // 2` of -1 needs floordiv), F6 (test inputs) |
| linked list | F7 (nodes → row pool with a sentinel), F9 (IndexError) |
| hash map | F7 + F8 (resize makes a new array, so `hm = hm_set(hm, k, v)`), F5 (`hash()`), F14 (negative keys need `pymod`), F9 (KeyError) |
| n-queens | F15 (printboard of `Q`/`.`), global solution list → returned counts |
| permutations | F2 (`append`/`pop` → overwrite position `index`), F15 |

## Friction ledger

**F1. `and`/`or` evaluate both sides in the interpreter, but only the needed side in C.** sev 3, engine gap, 10 guards rewritten
(a) had to write, in 10 places (merge, argsort, heapify ×2, max_heapify ×2, heappop, insertion sort, strip ×2):
```
while j > 0.0 {
    if insert_value >= a[j - 1.0] {
        break
    }
    ...
```
(b) wished: `while j > 0 and a[j - 1] > insert_value { ... }`
(c) Python and Julia (`&&`) both evaluate the right side only when needed.
`probe/and_evaluates_both.viv`: `if j >= 0.0 and a[j] > 0.0` with `j = -1` prints `7` natively
and stops with `index -1 out of bounds` in the interpreter. This breaks the byte-identical
promise. A program tested only natively can fail in the browser. Every bounds guard ends up nested
one level deeper.

**F2. No growable list (`append`/`push!`/`pop`).** sev 2, missing data structure, 13 sites
(a) Filters count first, then allocate and fill (`partition`, `members`, `knapsack_items`, the
sieve, `without`, `is_palindrome_ignore_case_and_spaces`). Or the port allocates the maximum
and trims (Kruskal's tree, copied row by row). Or it sizes capacity by hand (DFS stack =
edges + 1, heap/list/queue capacities):
```
count = 0.0
for i to len(s) { if is_alnum(s[i]) { count += 1.0 } }
kept = zeros(count)
k = 0.0
for i to len(s) { if is_alnum(s[i]) { kept[k] = s[i]; k += 1.0 } }
```
(b) `kept = [c for c in s if is_alnum(c)]` or `kept = filter(is_alnum, s)`, or `out = []; push(out, x)`.
(c) Python `[c for c in s if c.isalnum()]`, `list.append`; Julia `filter(isletter, s)`, `push!`.

**F3. No slice syntax, no list concatenation, no copy.** sev 2, missing builtin, 15 call sites + 3 helpers
(a) `seq.viv` defines `slice`, `copy` and `concat` as loops. `merge_sort(slice(collection, 0.0, mid))`,
`concat(concat(lesser, equal), greater)`, `is_palindrome_recursive(slice(s, 1.0, len(s) - 1.0))`.
(b) `collection[:mid]`, `lesser ++ equal ++ greater`, `s[1:-1]`.
(c) Python `a[:mid]`, `[*a, *b]`, `s[1:-1]`; Julia `a[1:mid]`, `vcat(a, b)`, `s[2:end-1]`.
`a[1:3]` is a parse error (`probe/slice_syntax.viv`).

**F4. Strings are arrays of character codes with no string library.** sev 2, strings/IO, 9 helpers in strings.viv, every string algorithm
(a) `chars("...")` makes a literal and `text(a)` prints one. Everything else is written by hand:
`lower`, `strip`, `without` (for `replace(" ", "")`), `reversed`, `is_alnum`, `is_space`,
`find`. A list of strings is not possible (F6), so the test data is one string with `|`
between words:
```
pal_words = chars("MALAYALAM|String|rotor|level|A|BB|ABC|amanaplanacanalpanama|abcdba|AB")
pal_at = word_starts(pal_words, BAR)
for k to 10 {
    s = word(pal_words, pal_at, k)
```
(b) `test_data = {"MALAYALAM": true, ...}` and `for (key, value) in test_data`, plus `s.lower()`, `s[::-1]`.
(c) Python `str` methods; Julia `lowercase`, `reverse`, `strip`, `replace`, `findfirst`.
`name = "hello"` is refused in both engines (`probe/string_value.viv`).

**F5. No dict, set or hash.** sev 2, missing data structure, 8 algorithms
(a) Letter keys become codes minus 65 (`chars("AAABBBCC...") - A`). Integer keys that are not
0..n-1 (Kahn's `{10: [20], 20: []}`) become a key list plus `index_of`, a linear search.
`defaultdict(int)` becomes `zeros(128)` indexed by code, which is ASCII only. A visited set
becomes a 0/1 mask. The hash map port writes Python's `hash()` for integers by hand:
```
def py_hash(key) {
    if key == -1.0 {
        return -2.0
    }
    return key
}
```
(b) `indegree = Dict(k => 0 for k in keys)`, `visited = Set()`.
(c) Python `dict`, `set`, `defaultdict`, `hash`; Julia `Dict`, `Set`, `hash`.
Dense integer keys as array indices work, and a mask is fine for BFS. The friction is in
sparse or non-numeric keys.

**F6. No ragged arrays (list of lists).** sev 2, missing data structure, 4 sites
(a) Adjacency lists become compressed rows built from an edge list. Three helpers plus a stable argsort:
```
dsrc = chars("AAABBBCCDDEEFFFG") - A
ddst = chars("BCDADEAFBDBFCEGF") - A
doff = row_offsets(7.0, dsrc)
dnbr = take(ddst, by_source(dsrc))
... for k to off[v + 1.0] - off[v] { adjacent = nbr[off[v] + k] ...
```
The heap class's 13 test lists become one flat array plus a lengths array.
(b) `G = [[1, 2, 3], [0, 3, 4], ...]` and `for adj in G[v]`.
(c) Python `dict[str, list[str]]`; Julia `Vector{Vector{Int}}`.
`[[1.0, 2.0], [3.0]]` is "matrix rows must all have the same length" (`probe/ragged_list.viv`).
CSR is a good representation, but writing it should not be a prerequisite for BFS.

**F7. Records do not run in the interpreter, and arrays of records do not compile to C.** sev 3, engine gap, 6 classes
(a) Every class becomes one array with header slots named by constants. Row 0 of the hash
map holds `(len, initial_block_size, capacity_factor)`, and bucket `b` is row `b + 1`. Row 0
of the linked list is `(length, rows used)`, row 1 is a sentinel node, and nodes are rows
`(data, next)`. The heap's item `i` is `h[H_ITEMS + i]`. Disjoint-set `Node`s become parallel
`parent`/`rank` arrays.
```
def list_insert_nth(ll: Arr, index, data) -> Bool {
    ...
    row = ll[0, 1]
    ll[0, 1] = row + 1.0
    prev = list_before(ll, index)
    ll[row, DATA] = data
    ll[row, NEXT] = ll[prev, NEXT]
```
(b) `record Node { data: Num, next: Node? }` or at least `record Stack { items: Arr, top: Num }` running in both engines.
(c) Python `class`/`@dataclass`; Julia `mutable struct`.
`probe/records.viv` runs natively and is refused by the interpreter. `probe/record_array.viv`
(`[e, e]` of an `Edge` record) is refused by C. The header-slot pattern works because arrays
alias like Python objects, but it is C-style struct packing. A wrong constant silently reads
another field.

**F8. A def returns one value.** sev 2, syntax, 5 sites
(a) `knapsack` returns the table and the caller reads `dp[n, w]`. LCS returns the
subsequence and the length is `len()`. Dijkstra's `heappop` moves the minimum to row
`size - 1` and returns the new size, and the caller reads `heap[size, 0]`, `heap[size, 1]`.
The hash map's `hm_set` returns the map because a resize replaces the array, so every call is
`hm = hm_set(hm, key, val)`.
(b) `cost, u = heappop(heap)`; `(n, seq) = lcs(x, y)`.
(c) Python tuples; Julia tuples.
`return 4.0, chars("GTAB")` is a parse error (`probe/tuple_return.viv`).

**F9. No exceptions, no `None`.** sev 2, control flow, 9 sites
(a) `stack_push` and `list_insert_nth` return `false` where Python raises. `stack_pop`,
`extract_max`, `list_delete_nth` and `hm_get` rely on the caller to check first. Kahn returns
an empty array for `None`. Python's input validation (`binary_search` on unsorted input,
`knapsack` weight types, sieve of a negative) is dropped.
(b) `error("Empty heap")` that a test can catch, and an optional/`nothing` result.
(c) Python `raise`/`try`; Julia `throw`/`try`, `nothing`.

**F10. A lambda passed to a def cannot capture a local in C (the interpreter allows it).** sev 2, engine gap, 1 site (3 comprehensions)
(a) quick sort's `[x for x in c if x < pivot]` (and `==`, `>`) became one def with a mode flag:
```
def partition(a: Arr, pivot, side) -> Arr {
    ... if sign(a[i] - pivot) == side { ...
lesser = quick_sort(partition(collection, pivot, -1.0))
```
(b) `lesser = select(collection, fn(x) -> x < pivot)`.
(c) Python comprehension; Julia `filter(x -> x < pivot, c)`.
`probe/closure_capture.viv` runs in the interpreter and is refused by C.

**F11. No nested defs.** sev 1, syntax, 3 sites
(a) `merge`, Kruskal's `find_parent` and `_binary_search_recursive` moved to top level, with
the captured variables (`parent`, `sorted_collection`, `item`) added as parameters.
(b) a `def` inside a `def`. (c) Python nested functions; Julia inner functions.
`probe/nested_def.viv`.

**F12. `+=` on an array element is refused.** sev 1, syntax, 18 sites
(a) `indegree[neighbor] = indegree[neighbor] - 1.0`, `off[src[e] + 1.0] = off[src[e] + 1.0] + 1.0`.
(b) `indegree[neighbor] -= 1`. (c) the same in Python and Julia.
The error message names the fix. `+=` on a scalar works.

**F13. `for i to n` is the only range.** sev 1, control flow, 16 loops
(a) Descending and strided loops are `while`: heap_sort's `range(n // 2 - 1, -1, -1)`, DFS's
`reversed(graph[v])`, the sieve's `range(p * p, num + 1, p)`, and offsets like `for i0 to n { i = i0 + 1.0 ...`.
(b) `for i in n/2-1 down to 0`, `for i from p*p to num step p`.
(c) Python `range(a, b, step)`, `reversed`; Julia `n:-1:1`, `p*p:p:num`.

**F14. One number type: no integers, no big integers, and `mod` is C's fmod.** sev 3, type system, 19 `pymod`/`floordiv` sites + 2 wrong results
(a) `pymod` and `floordiv` in `seq.viv` are needed wherever a negative number meets `%` or
`//`. The hash map's bucket for key -7 in 10 buckets is `-7 % 10 == 3` in Python; fmod gives
-7, an out-of-bounds index. The heap's `(0 - 1) // 2` must be -1. gcd uses `pymod` to follow
the original; its doctests happen to pass with fmod too, because of the final `abs`
(`probe/gcd_with_fmod.viv`). Anything above 2^53 is
inexact. `binary_exp_mod_iterative(1269380576, 374, 34)` gives `nan` (Python: 8) and
`binary_exp_iterative(1269380576, 374)` gives `inf` (Python: 3,400 digits). `band`/`shr`
exist but are 32-bit, so `exponent & 1` and `exponent >>= 1` were written as `pymod`/`floordiv`.
(b) an `Int` type with `%` and `//` defined Python's way, and ideally big integers.
(c) Python `int` (arbitrary precision); Julia `Int64`, `mod`, `div`, `BigInt`.
`probe/fmod_sign.viv`, `probe/big_integers.viv`. Where fmod and Python's `%` disagree, the
result is a wrong number or an out-of-bounds index with no hint that `%` was the cause.

**F15. Output: `show` labels only names, `print` takes only scalars, numbers print at 6 digits, text cannot be mixed in.** sev 2, strings/IO, all output (≈75 `show` lines)
(a) Every printed value is first assigned to a name so `show` labels it
(`kmp_1 = knuth_morris_pratt(...)` then `show kmp_1`). `11 ** 13` prints as `3.45227e+13`.
`text()` always prints `text: ...` with no label. n-queens' `printboard` (`Q . . .`)
became a 0/1 matrix.
(b) `println("len = $n, sub-sequence = $(text(seq))")`; `print(a)` for arrays.
(c) Python f-strings; Julia string interpolation, `@printf`.
`probe/print_array.viv`.

**F16. Arrays cannot hold `Bool`.** sev 1, type system, ≈27 `== 1.0`/`== 0.0` tests
(a) `visited[v] == 1.0`, `primes[p] == 1.0`, `if is_palindrome(s) { palindromes[k, 0] = 1.0 }`.
(b) `visited = falses(n)`, `if visited[v]`.
(c) Python `[False] * n`; Julia `falses(n)`, `BitVector`.
`probe/bool_array.viv`.

**F17. No standard containers or sort by key.** sev 2, missing builtin, 4 sites
(a) Dijkstra needs a binary min-heap with tuple ordering (`less`, `swap_rows`, `heappush`,
`heappop`: 45 lines). Kruskal's `sorted(edges, key=lambda e: e[2])` needs a stable
`argsort` (33 lines in `seq.viv`). `deque`/`Queue` become a head and tail index. `str.find`
is written by hand for the KMP check.
(b) `#! use lib/heap` / `argsort(keys)` / `sort(a, by = ...)`.
(c) Python `heapq`, `collections.deque`, `sorted(key=)`; Julia `DataStructures.jl`, `sortperm`.

**F18. The interpreter crashes on recursion 1000–2000 deep, and native C does not.** sev 2, engine gap, 0 sites in the ports, found by probe
(a) nothing in these ports hit it, because the inputs are small. `probe/deep_recursion.viv`
at depth 2000 prints `2000` natively and aborts the interpreter with `thread 'main' has
overflowed its stack` and no location. A recursive DFS on a 2,000-node path graph, or
`find_set` before path compression on a long chain, would do the same.
(b) a clean `error: recursion deeper than N` with the call site, and the same limit in every engine.
(c) Python raises `RecursionError` at depth 1000; Julia's limit is the C stack, much deeper.

**F19. Error locations are wrong in a program that uses a module.** sev 2, tooling/error message, hit once
(a) The first `graphs.viv` compile error was reported at `graphs.viv:121` for a line at 19.
`probe/use_line_number.viv` (`#! use lib/params`, error on line 3) reports line 206, for a
compile error and a run-time error alike (`probe/use_line_runtime.out`). The number is offset
by the length of the module. The quoted source line is correct.
(b) the program's own line number. (c) n/a.

**F20. `chars("")` is refused.** sev 1, strings/IO, 7 test sites
(a) the empty string is `zeros(0.0)`: `longest_common_subsequence(zeros(0.0), chars("abc"))`.
(b) `chars("")`. (c) `""`.

**F21. `==` does not compare arrays.** sev 1, missing builtin, 50 `same()` calls
(a) `same(a, b)` in `seq.viv`, and `assert_near(x, 3.0, 0.0)` for integer equality.
(b) `assert_true(a == b)` / `assert_eq(a, b)`. (c) Python `==` on lists; Julia `==` on arrays.
`probe/array_equality.viv`. `assert_allclose(a, b, 0.0)` works in tests but is not a Bool.

**F22. `chars` and `text` are not in the reference page or its builtin table.** sev 1, tooling, 1
(a) found by reading `lang/viv/src/parser.rs` (`chars` desugars in the parser) and `interp.rs`
(`("text", 1)`). Before that, the reference page's "strings are not values" implied that
strings could not be ported.
(b) both listed under Builtins, with the one-literal restriction. (c) n/a.

## Wins

- **Three engines, one output.** Every port prints byte-identical output in the interpreter, native C and the direct wasm backend (node), and all 25 tests pass in each.
- **The algorithms read like the Python where the data is numbers.** Merge sort's recursion, heapify, path-compressed `find_set`, memoized knapsack and edit distance, and n-queens' place/recurse/undo are line-for-line transcriptions.
- **Rank-2 arrays make DP tables cleaner than lists of lists.** `dp = zeros(m + 1.0, n + 1.0)` and `dp[i - 1.0, j - 1.0]` replace `[[0] * (n + 1) for _ in range(m + 1)]` and `dp[i-1][j-1]`. A zero-width table works, so edit distance against `""` needs no special case (`zeros(9, 0)`).
- **Arrays alias the way Python objects do.** A def that writes into its array argument changes the caller's structure, the way a method changes `self`. Stack, queue, heap and linked list operations are in-place defs with no rebinding.
- **Whole-array operations remove loops.** `costs = zeros(n) + INF` (with `INF = 1.0 / 0.0`), `chars("AAB...") - A` to turn letters into vertex numbers, `sum(mask)` to count a set, `take(dst, argsort(src))` to group edges by source, `slab(transpose(edges), 2.0)` to get a column, `setslab` to copy a row, and `linspace(-8.0, 0.0, 9.0)` for an expected range.
- **Tests are short.** `assert_allclose(mst2, [[2.0, 3.0, 1.0], [0.0, 2.0, 1.0], [0.0, 1.0, 3.0]], 0.0)` checks a whole doctest in one line, and each test block prints one PASS line.
- **`chars`/`text` make string algorithms possible.** KMP, LCS and edit distance on code arrays compare characters with `==` as in Python, and `text(lcs)` prints `GTAB`.
- **Error messages point at the fix.** Compound assignment says to write `a[i] = a[i] + ...`. A negative index notes that there is no negative indexing. A string value says where strings are allowed.
- **Speed.** Every program finishes in under 0.1 s in the interpreter.
