# lib/algo/algo.spec: what each algorithm of lib/algo promises, as data.
#
# tools/algo/check.py reads this file: the `post` lines of an
# `algo` are its postconditions, which the harness checks on generated inputs (in Viv,
# lib/algo/props.viv, in all three engines, and again in Python); `bound` is the
# complexity cppreference states, which the harness measures where it can count; `oracle`
# names the Python or NumPy function the results must equal byte for byte. An `impl` is an
# implementation that exists today (a builtin, or a corpus copy in lib/algo/legacy) and
# how to run it.
#
# Blocks start at column 0; fields are indented two spaces; a `call` line is Viv, taken
# as written (four more spaces of indentation inside braces). `#` starts a comment line.
#
# Variables of a case: X is the input (lib/algo/cases.viv's algo_case(kind, n, seed));
# the call lines set the others. Kinds: random dups sorted reversed equal organ sawtooth
# nearly zeros special (cases.viv says what each is).

order total  the order `sort` puts values in
  what  ascending; -0.0 before 0.0; NaN after every number (lower.rs lower_sort, interp.rs sort_order)

order numpy  the order of every comparison-based algorithm in lib/algo and of `argsort`
  what  ascending; -0.0 equal to 0.0; NaN after every number and equal to every NaN (np.sort)

# ---------------------------------------------------------------- tier 1: needs Int

algo sort  sort an array, in place
  tier     1
  sig      sort(a: [Num])
  effect   in place: a
  order    total
  post     permutation R X
  post     ordered-total R
  stable   unobservable: the total order leaves no two different doubles equal
  bound    comparisons  n log n  "𝓞(N·log(N)) comparisons"
  cite     https://en.cppreference.com/w/cpp/algorithm/sort
  oracle   R np-sort-total X
  members  builtin introsort radix heapsort merge
  default  builtin

algo sort_by  sort an array of records, or of values, by a key, stably
  tier     1
  sig      sort_by(a: [Num], key: [Num])
  effect   in place: a (and the key, moved with it)
  order    numpy
  post     permutation R X
  post     ordered R
  post     stable-values R X
  bound    comparisons  n log n  "𝓞(N·log2(N)) comparisons ... (or only 𝓞(N·log(N)) comparisons if enough extra memory is available)"
  cite     https://en.cppreference.com/w/cpp/algorithm/stable_sort
  oracle   R np-sort-stable X
  members  merge timsort radix
  default  merge

algo argsort  the positions that put an array in order, ties in input order
  tier     1
  sig      argsort(a: [Num]) -> [Int]
  effect   returns a new array
  order    numpy
  post     index-perm P X
  post     ordered-by P X
  post     stable-by P X
  bound    comparisons  n log n  "𝓞(N·log2(N)) comparisons" (stable_sort)
  cite     https://en.cppreference.com/w/cpp/algorithm/stable_sort
  oracle   P np-argsort-stable X
  members  builtin radix
  default  builtin

algo partial_sort  the k smallest entries, in order, at the front
  tier     1
  sig      partial_sort(a: [Num], k: Int)
  effect   in place: a
  order    numpy
  post     permutation R X
  post     partial-sorted R X K
  bound    comparisons  n log k  "Approximately N·log(M) comparisons"
  cite     https://en.cppreference.com/w/cpp/algorithm/partial_sort
  members  heap_select quickselect_then_sort
  default  heap_select

algo nth_element  the entry that would be at position k in sorted order, at k; smaller ones before, larger after
  tier     1
  sig      nth_element(a: [Num], k: Int)
  effect   in place: a
  order    numpy
  post     permutation R X
  post     nth R X K
  bound    comparisons  n  "𝓞(N) comparisons ... on average"
  cite     https://en.cppreference.com/w/cpp/algorithm/nth_element
  oracle   R np-partition X K
  members  introselect quickselect median_of_medians
  default  introselect

algo is_sorted  whether no entry comes before the one in front of it
  tier     1
  sig      is_sorted(a: [Num]) -> Bool
  effect   reads only
  order    numpy
  post     is-sorted-answer F X
  bound    comparisons  exact n - 1  "𝓞(N) comparisons"
  cite     https://en.cppreference.com/w/cpp/algorithm/is_sorted

algo lower_bound  the first position in a sorted array whose entry does not come before x
  tier     1
  sig      lower_bound(v: [Num], x: Num) -> Int
  effect   reads only
  order    numpy
  post     lower-bounds RS V Q
  bound    comparisons  log n  "At most log2(N)+𝓞(1) comparisons with value"
  cite     https://en.cppreference.com/w/cpp/algorithm/lower_bound
  oracle   RS np-searchsorted-left V Q

algo upper_bound  the first position in a sorted array whose entry comes after x
  tier     1
  sig      upper_bound(v: [Num], x: Num) -> Int
  effect   reads only
  order    numpy
  post     upper-bounds RS V Q
  bound    comparisons  log n  "At most log2(N)+𝓞(1) comparisons with value"
  cite     https://en.cppreference.com/w/cpp/algorithm/upper_bound
  oracle   RS np-searchsorted-right V Q

algo equal_range  lower_bound and upper_bound together
  tier     1
  sig      equal_range(v: [Num], x: Num) -> (Int, Int)
  effect   reads only
  order    numpy
  post     lower-bounds LO V Q
  post     upper-bounds HI V Q
  bound    comparisons  log n  "At most 2log2(N)+𝓞(1) comparisons with value"
  cite     https://en.cppreference.com/w/cpp/algorithm/equal_range

algo searchsorted  lower_bound or upper_bound of many values at once (NumPy's name and side=)
  tier     1
  sig      searchsorted(v: [Num], xs: [Num], side: Side) -> [Int]
  effect   returns a new array
  order    numpy
  post     lower-bounds RS V Q
  bound    comparisons  log n  "At most log2(N)+𝓞(1) comparisons with value" (per value, as lower_bound)
  cite     https://en.cppreference.com/w/cpp/algorithm/lower_bound
  oracle   RS np-searchsorted-left V Q

algo binary_search  whether a sorted array holds x
  tier     1
  sig      binary_search(v: [Num], x: Num) -> Bool
  effect   reads only
  order    numpy
  post     contains RS V Q
  bound    comparisons  log n  "At most log2(N)+𝓞(1) comparisons with value"
  cite     https://en.cppreference.com/w/cpp/algorithm/binary_search

algo unique  the distinct values of an array, sorted (np.unique)
  tier     1
  sig      unique(a: [Num]) -> [Num]
  effect   returns a new array
  order    numpy
  post     unique-sorted R X
  bound    comparisons  n log n  "Exactly max(0,N-1) comparisons using operator==" (std::unique, after a sort)
  cite     https://en.cppreference.com/w/cpp/algorithm/unique
  oracle   R np-unique X

algo unique_runs  each run of equal neighbors kept once (std::unique)
  tier     1
  sig      unique_runs(a: [Num]) -> [Num]
  effect   returns a new array
  order    numpy
  post     unique-runs R X
  bound    comparisons  exact n - 1  "Exactly max(0,N-1) comparisons using operator=="
  cite     https://en.cppreference.com/w/cpp/algorithm/unique

algo partition  the entries that satisfy p first, then the rest; returns where the second group starts
  tier     1
  sig      partition(a: [Num], p: fn(Num) -> Bool) -> Int
  effect   in place: a
  post     permutation R X
  post     partitioned R K
  bound    predicate  exact n  "At most N swaps ..., and exactly N applications of p."
  cite     https://en.cppreference.com/w/cpp/algorithm/partition
  members  hoare lomuto
  default  hoare

algo stable_partition  partition, each group in input order
  tier     1
  sig      stable_partition(a: [Num], p: fn(Num) -> Bool) -> Int
  effect   in place: a
  post     stable-partition R X K
  bound    predicate  exact n  "At most N·log2(N) swaps (or only 𝓞(N) swaps if there is enough extra memory), and exactly N applications of p."
  cite     https://en.cppreference.com/w/cpp/algorithm/stable_partition

algo partition_point  where the entries satisfying p end, in a partitioned array
  tier     1
  sig      partition_point(a: [Num], p: fn(Num) -> Bool) -> Int
  effect   reads only
  post     partition-point K X
  bound    predicate  log n  "performs 𝓞(log(N)) applications of the predicate p"
  cite     https://en.cppreference.com/w/cpp/algorithm/partition_point

algo reverse  the entries in the opposite order
  tier     1
  sig      reverse(a: [Num])
  effect   in place: a
  post     reversed R X
  bound    swaps  exact n / 2  "Exactly std::distance(first, last) / 2 swaps."
  cite     https://en.cppreference.com/w/cpp/algorithm/reverse
  oracle   R reversed X

algo rotate  the entries shifted left by k, the first k moved to the end
  tier     1
  sig      rotate(a: [Num], k: Int)
  effect   in place: a
  post     rotated R X K
  bound    swaps  n  "At most std::distance(first, last) swaps."
  cite     https://en.cppreference.com/w/cpp/algorithm/rotate
  oracle   R np-roll-left X K

algo min_element  the first position of the smallest entry
  tier     1
  sig      min_element(a: [Num]) -> Int
  effect   reads only
  order    numpy
  post     min-pos R X
  bound    comparisons  exact n - 1  "Exactly max(N-1,0) comparisons"
  cite     https://en.cppreference.com/w/cpp/algorithm/min_element

algo max_element  the first position of the largest entry
  tier     1
  sig      max_element(a: [Num]) -> Int
  effect   reads only
  order    numpy
  post     max-pos R X
  bound    comparisons  exact n - 1  "Exactly max(N-1,0) comparisons"
  cite     https://en.cppreference.com/w/cpp/algorithm/max_element

algo merge  two sorted arrays as one sorted array, ties from the first array first
  tier     1
  sig      merge(a: [Num], b: [Num]) -> [Num]
  effect   returns a new array
  order    numpy
  post     merged R A B
  bound    comparisons  exact n1 + n2 - 1  "At most N1+N2-1 comparisons"
  cite     https://en.cppreference.com/w/cpp/algorithm/merge

algo set_union  the union of two sorted arrays, counting repeats (max of the two counts)
  tier     1
  sig      set_union(a: [Num], b: [Num]) -> [Num]
  effect   returns a new array
  order    numpy
  post     set-union R A B
  bound    comparisons  exact 2 (n1 + n2) - 1  "At most 2·(N1+N2)-1 comparisons"
  cite     https://en.cppreference.com/w/cpp/algorithm/set_union

algo set_intersection  the common entries of two sorted arrays (min of the two counts)
  tier     1
  sig      set_intersection(a: [Num], b: [Num]) -> [Num]
  effect   returns a new array
  order    numpy
  post     set-intersection R A B
  bound    comparisons  exact 2 (n1 + n2) - 1  "At most 2·(N1+N2)-1 comparisons"
  cite     https://en.cppreference.com/w/cpp/algorithm/set_intersection

algo set_difference  the entries of the first sorted array not matched in the second
  tier     1
  sig      set_difference(a: [Num], b: [Num]) -> [Num]
  effect   returns a new array
  order    numpy
  post     set-difference R A B
  bound    comparisons  exact 2 (n1 + n2) - 1  "At most 2·(N1+N2)-1 comparisons"
  cite     https://en.cppreference.com/w/cpp/algorithm/set_difference

algo set_symmetric_difference  the entries in one sorted array and not matched in the other
  tier     1
  sig      set_symmetric_difference(a: [Num], b: [Num]) -> [Num]
  effect   returns a new array
  order    numpy
  post     set-symmetric-difference R A B
  bound    comparisons  exact 2 (n1 + n2) - 1  "At most 2·(N1+N2)-1 comparisons"
  cite     https://en.cppreference.com/w/cpp/algorithm/set_symmetric_difference

algo includes  whether every entry of the second sorted array is matched in the first
  tier     1
  sig      includes(a: [Num], b: [Num]) -> Bool
  effect   reads only
  order    numpy
  post     includes F A B
  bound    comparisons  exact 2 (n1 + n2) - 1  "At most 2·(N1+N2)-1 comparisons"
  cite     https://en.cppreference.com/w/cpp/algorithm/includes

algo next_permutation  the next arrangement in lexicographic order; false (and sorted) after the last
  tier     1
  sig      next_permutation(a: [Num]) -> Bool
  effect   in place: a
  order    numpy
  post     next-perm R F X
  bound    swaps  n / 2  "At most N/2 swaps."
  cite     https://en.cppreference.com/w/cpp/algorithm/next_permutation
  oracle   R itertools-next X

algo is_permutation  whether two arrays hold the same values as often each
  tier     1
  sig      is_permutation(a: [Num], b: [Num]) -> Bool
  effect   reads only
  post     is-permutation-answer F A B
  bound    comparisons  n^2  "𝓞(N^2) comparisons using operator== (or only exactly N comparisons if the two target ranges are lexicographically equal)"
  cite     https://en.cppreference.com/w/cpp/algorithm/is_permutation

algo gcd  the greatest common divisor, never negative; gcd(0, 0) = 0
  tier     1
  sig      gcd(a: Int, b: Int) -> Int
  effect   reads only
  post     gcd R A B
  bound    steps  log n  (cppreference states none; Euclid takes at most about 2.08 ln(n) + 1.67 steps, Knuth TAOCP 4.5.3)
  cite     https://en.cppreference.com/w/cpp/numeric/gcd
  oracle   R math-gcd A B

algo lcm  the least common multiple, never negative; 0 when either is 0
  tier     1
  sig      lcm(a: Int, b: Int) -> Int
  effect   reads only
  post     lcm R A B
  bound    steps  log n  (cppreference states none)
  cite     https://en.cppreference.com/w/cpp/numeric/lcm
  oracle   R math-lcm A B

# ---------------------------------------------------------------- tier 2: needs arrays of records

algo heap  make_heap, push_heap, pop_heap, sort_heap; a priority queue on top
  tier     2
  sig      heap_push(h: Heap, x: Num), heap_pop(h: Heap) -> Num, with Heap a record holding an array
  effect   in place: h
  order    numpy
  post     max-heap R
  post     permutation R X
  bound    comparisons  exact 3 n  "At most 3N comparisons" (make_heap); "At most log(N)" (push_heap); "At most 2log(N)" (pop_heap)
  cite     https://en.cppreference.com/w/cpp/algorithm/make_heap

algo pop_order  the entries popped from a heap, one by one
  tier     2
  sig      heap_pop(h: Heap) -> Num
  effect   in place: h
  order    numpy
  post     ordered-desc R
  post     permutation R X
  bound    comparisons  n log n  "At most 2N·log(N) comparisons" (sort_heap)
  cite     https://en.cppreference.com/w/cpp/algorithm/sort_heap
  oracle   R sorted-desc X

algo min_heap_rows  a min-heap of (cost, vertex) rows, popped in the order heapq pops tuples
  tier     2
  sig      a priority queue of records { cost: Num, vertex: Int }
  effect   in place
  post     ordered R
  bound    comparisons  n log n  "At most log(N)" per push_heap, "At most 2log(N)" per pop_heap
  cite     https://en.cppreference.com/w/cpp/algorithm/push_heap
  oracle   R heapq-pops X

algo deque  a double-ended queue
  tier     2
  sig      a record { items: [Num], head: Int, count: Int }
  bound    operations  1  amortized constant push and pop at either end (std::deque)
  cite     https://en.cppreference.com/w/cpp/container/deque

algo union_find  disjoint sets with union by rank and path compression
  tier     2
  sig      a record { parent: [Int], rank: [Int] }
  bound    operations  α(n)  amortized inverse Ackermann per operation (Tarjan 1975)

algo sort_records  sort an array of records by a field, stably
  tier     2
  sig      sort_by(rs: [R], key: fn(R) -> Num)
  effect   in place
  post     stable-by P X
  bound    comparisons  n log n  "𝓞(N·log2(N)) comparisons" (stable_sort)
  cite     https://en.cppreference.com/w/cpp/algorithm/stable_sort

# ---------------------------------------------------------------- tier 3

algo topological_sort  the vertices so that every edge goes forward (Kahn)
  tier     3
  sig      topological_sort(g: Graph) -> [Int]?
  effect   returns a new array, or None on a cycle
  post     topological R K SRC DST
  bound    operations  n + m  (Kahn 1962: each vertex and edge once)
  oracle   R kahn K SRC DST

algo bfs  breadth-first search: the vertices in order of distance
  tier     3
  sig      bfs(g: Graph, start: Int) -> [Int]
  bound    operations  n + m

algo dfs  depth-first search
  tier     3
  sig      dfs(g: Graph, start: Int) -> [Int]
  bound    operations  n + m

algo dijkstra  shortest paths from one vertex, nonnegative weights
  tier     3
  sig      dijkstra(g: Graph, start: Int) -> [Num]
  bound    operations  (n + m) log n  (with a binary heap)

algo mst  minimum spanning tree (Kruskal with union_find; Prim with a heap)
  tier     3
  sig      mst(g: Graph) -> [Int]
  bound    operations  m log m

algo kmp  the positions of a pattern in a string (Knuth-Morris-Pratt)
  tier     3
  sig      find_all(s: Str, p: Str) -> [Int]
  bound    comparisons  n + m

algo edit_distance  Levenshtein distance
  tier     3
  sig      edit_distance(a: Str, b: Str) -> Int
  bound    operations  n m

algo fenwick  prefix sums with point updates
  tier     3
  sig      a record { tree: [Num] }; fen_add(f, i, x), fen_sum(f, i) -> Num
  bound    operations  log n  per update or query

algo intervals  merge overlapping intervals; the largest set of disjoint ones
  tier     3
  sig      merge_intervals(lo: [Num], hi: [Num]) -> ([Num], [Num])
  bound    comparisons  n log n

# ---------------------------------------------------------------- tier 4

algo generic  every algorithm above for any element type with an order
  tier     4
  sig      sort(a: [T]) where T: Ordered; lower_bound(v: [T], x: T) -> Int; ...

# ================================================================ implementations that exist today

impl builtin_sort  the `sort` builtin
  algo     sort
  module   builtin
  call     R = copy(X)
  call     sort(R)
  note     stable-values R X
  oracle   R np-sort-total X
  oracle   R np-sort-stable X  info

impl phrase_sorted  the phrase `take V sorted` (phrases.rs: take(V, argsort(V), 0))
  algo     sort
  module   builtin
  call     R = take X sorted
  post     permutation R X
  post     ordered R
  post     stable-values R X
  note     ordered-total R
  oracle   R np-sort-stable X

impl builtin_argsort  the `argsort` builtin
  algo     argsort
  module   builtin
  call     P = argsort(X)

impl builtin_argsort_key  `argsort(a, key=...)`, which parser.rs lowers to a merge sort in Viv
  algo     argsort
  module   builtin
  call     P = argsort(X, key=fn(x) -> x)

impl builtin_argmin  the `argmin` builtin
  algo     min_element
  module   builtin
  call     R = argmin(X)
  known    min-pos special  "argmin returns the first NaN, as np.argmin does; in the NaN-last order the smallest entry is the smallest number"
  sizes    1 2 3 4 5 7 8 9 15 16 17 31 32 33 63 64 65 129 257
  oracle   R np-argmin X

impl builtin_argmax  the `argmax` builtin
  algo     max_element
  module   builtin
  call     R = argmax(X)
  sizes    1 2 3 4 5 7 8 9 15 16 17 31 32 33 63 64 65 129 257
  oracle   R np-argmax X

impl ta_merge_sort  TheAlgorithms merge_sort (study/idioms/thealgorithms-python/sorting.viv)
  algo     sort_by
  module   lib/algo/legacy/ta_sorting
  call     R = merge_sort(X)
  outside  special  "compares with `<`, which is false for NaN; the port promises nothing for NaN"
  count    lg_lt
  adversary yes

impl ta_quick_sort  TheAlgorithms quick_sort: random pivot, three lists
  algo     sort_by
  module   lib/algo/legacy/ta_sorting
  call     R = quick_sort(X)
  outside  special  "compares by the sign of a difference; the port promises nothing for NaN"

impl ta_heap_sort  TheAlgorithms heap_sort
  algo     sort_by
  module   lib/algo/legacy/ta_sorting
  call     R = heap_sort(copy(X))
  post     permutation R X
  post     ordered R
  note     stable-values R X
  oracle   R np-sort-stable X  info
  outside  special  "compares with `<`"
  count    lg_lt
  adversary yes

impl ta_insertion_sort  TheAlgorithms insertion_sort
  algo     sort_by
  module   lib/algo/legacy/ta_sorting
  call     R = insertion_sort(copy(X))
  outside  special  "compares with `<`"
  count    lg_lt
  growth   n^2  "insertion sort: N(N-1)/2 comparisons on reversed input"

impl ta_binary_search  TheAlgorithms binary_search: the leftmost index of x, or -1
  algo     binary_search
  module   lib/algo/legacy/ta_sorting
  call     V = copy(X)
  call     sort(V)
  call     Q = algo_needles(V)
  call     RS = zeros(len(Q))
  call     for k to len(Q) {
  call         RS[k] = binary_search(V, Q[k])
  call     }
  post     first-or-none RS V Q
  oracle   RS first-index V Q
  outside  special  "NaN needles and NaN entries compare false"
  count    lg_lt lg_eq
  per      Q
  growth   2 log n  "an == and a < per halving, where std::binary_search makes one < per halving and one more at the end"

impl ta_bisect_left  TheAlgorithms bisect_left
  algo     lower_bound
  module   lib/algo/legacy/ta_sorting
  call     V = copy(X)
  call     sort(V)
  call     Q = algo_needles(V)
  call     RS = zeros(len(Q))
  call     for k to len(Q) {
  call         RS[k] = bisect_left(V, Q[k])
  call     }
  oracle   RS bisect-left V Q
  outside  special  "compares with `<`"
  count    lg_lt
  per      Q

impl ta_binary_search_rec  TheAlgorithms binary_search_by_recursion
  algo     binary_search
  module   lib/algo/legacy/ta_sorting
  call     V = copy(X)
  call     sort(V)
  call     Q = algo_needles(V)
  call     RS = zeros(len(Q))
  call     for k to len(Q) {
  call         RS[k] = binary_search_by_recursion(V, Q[k])
  call     }
  post     first-or-none RS V Q
  oracle   RS first-index V Q
  outside  special  "compares with `<` and `==`"

impl seq_argsort  the argsort every TheAlgorithms port uses (seq.viv), shadowing the builtin
  algo     argsort
  module   lib/algo/legacy/seq_argsort
  call     P = seq_argsort(X)
  outside  special  "compares with `<=`"
  count    lg_le
  adversary yes

impl cmp_merge_sort  the /compare page's merge sort on lists
  algo     sort_by
  module   lib/algo/legacy/cmp_merge_sort
  call     R = merge_sort(X)
  outside  special  "compares with `<=`"
  count    lg_le
  adversary yes

impl cmp_quicksort  the /compare page's quicksort (Lomuto, last element as pivot)
  algo     sort_by
  module   lib/algo/legacy/cmp_quicksort
  call     R = copy(X)
  call     quicksort(R, 0, len(R) - 1)
  post     permutation R X
  post     ordered R
  note     stable-values R X
  oracle   R np-sort-stable X  info
  outside  special  "compares with `<`"
  count    lg_lt
  known    growth dups sorted reversed equal organ adversary  "Lomuto partition around the last element: sorted, reversed and equal inputs make every partition empty on one side (n(n-1)/2 comparisons)"
  adversary yes

impl cmp_insertion_sort  the /compare page's insertion sort
  algo     sort_by
  module   lib/algo/legacy/cmp_insertion_sort
  call     R = insertion_sort(X)
  outside  special  "compares with `>`"
  count    lg_gt
  growth   n^2  "insertion sort"

impl bench_qsort  the Julia microbenchmark's quicksort (middle element as pivot)
  algo     sort_by
  module   lib/algo/legacy/bench_qsort
  call     R = copy(X)
  call     if len(R) > 0 {
  call         qsort(R, 0, len(R) - 1)
  call     }
  post     permutation R X
  post     ordered R
  note     stable-values R X
  oracle   R np-sort-stable X  info
  outside  special  "compares with `<` and `>`"
  count    lg_lt lg_gt
  known    growth organ adversary  "the middle element as pivot: organ-pipe input puts the largest value in the middle at every level, and McIlroy's adversary builds such inputs for any fixed pivot rule"
  adversary yes

impl bench_issorted  the Julia microbenchmark's issorted
  algo     is_sorted
  module   lib/algo/legacy/bench_qsort
  call     F = issorted(X)
  outside  special  "compares with `>`"
  oracle   F python-is-sorted X

impl eig_sort  lib/hf's eig_sort: selection sort of eigenvalues, moving each eigenvector's column with it
  algo     sort_by
  module   lib/algo/legacy/eig_sort
  call     D = copy(X)
  call     W = eye(len(X))
  call     eig_sort(D, W, len(X))
  call     P = zeros(len(X))
  call     for j to len(X) {
  call         for i to len(X) {
  call             if W[i, j] == 1.0 {
  call                 P[j] = i
  call             }
  call         }
  call     }
  post     ordered D
  post     index-perm P X
  post     follows D P X
  note     stable-values D X
  oracle   D np-sort-stable X  info
  outside  special  "compares with `<`; eigenvalues of a symmetric matrix are never NaN"
  sizes    0 1 2 3 4 5 7 8 9 15 16 17 31 32 33 63 64 65
  count    lg_lt
  growth   n^2  "selection sort: N(N-1)/2 comparisons on every input"

impl npr_lg_search  lib/nprandom's searchsorted(side='right')
  algo     upper_bound
  module   lib/algo/legacy/searches
  call     V = copy(X)
  call     sort(V)
  call     Q = algo_needles(V)
  call     RS = zeros(len(Q))
  call     for k to len(Q) {
  call         RS[k] = npr_lg_search(V, Q[k])
  call     }
  outside  special  "a cumulative distribution has no NaN"
  count    lg_le
  per      Q

impl fasta_bisect  the fasta benchmark's bisect.bisect
  algo     upper_bound
  module   lib/algo/legacy/searches
  call     V = copy(X)
  call     sort(V)
  call     Q = algo_needles(V)
  call     RS = zeros(len(Q))
  call     for k to len(Q) {
  call         RS[k] = bisect(V, Q[k])
  call     }
  oracle   RS bisect-right V Q
  outside  special  "compares with `<`"
  count    lg_lt
  per      Q

impl ta_build_max_heap  TheAlgorithms heap.py: build_max_heap
  algo     heap
  module   lib/algo/legacy/heaps
  call     H = heap_new(len(X))
  call     build_max_heap(H, X)
  call     R = heap_items(H)
  outside  special  "compares with `>`"
  count    lg_gt
  growth   exact 3 n  "make_heap: at most 3N comparisons"

impl ta_heap_insert  TheAlgorithms heap.py: insert one by one
  algo     heap
  module   lib/algo/legacy/heaps
  call     H = heap_new(len(X))
  call     for i to len(X) {
  call         heap_insert(H, X[i])
  call     }
  call     R = heap_items(H)
  outside  special  "compares with `>`"
  count    lg_gt
  growth   n log n  "N pushes of at most log(N) comparisons each"

impl ta_extract_max  TheAlgorithms heap.py: build, then extract_max until empty
  algo     pop_order
  module   lib/algo/legacy/heaps
  call     H = heap_new(len(X))
  call     build_max_heap(H, X)
  call     R = zeros(len(X))
  call     for i to len(X) {
  call         R[i] = extract_max(H)
  call     }
  outside  special  "compares with `>`"
  oracle   R sorted-desc X  info
  count    lg_gt

impl ta_heap_sort_in_place  TheAlgorithms heap.py: heap_sort_in_place
  algo     sort_by
  module   lib/algo/legacy/heaps
  call     H = heap_new(len(X))
  call     build_max_heap(H, X)
  call     heap_sort_in_place(H)
  call     R = heap_items(H)
  post     permutation R X
  post     ordered R
  note     stable-values R X
  oracle   R np-sort-stable X  info
  outside  special  "compares with `>`"

impl dijkstra_heap  the (cost, vertex) min-heap of TheAlgorithms' Dijkstra port
  algo     min_heap_rows
  module   lib/algo/legacy/heaps
  call     H = zeros(len(X) + 1, 2)
  call     size = 0
  call     for i to len(X) {
  call         size = heappush(H, size, X[i], i)
  call     }
  call     R = zeros(len(X))
  call     for i to len(X) {
  call         size = heappop(H, size)
  call         R[i] = H[size, 0]
  call     }
  outside  special  "compares with `<` and `!=`"
  count    lg_lt lg_ne
  growth   n log n  "push_heap: at most log(N); pop_heap: at most 2log(N)"

impl gcd_post  the gcd of posts/2026-09-25-the-error-follows-the-condition-number.viv
  algo     gcd
  module   lib/algo/legacy/gcd_post
  call     A = floor(X * 200)
  call     B = floor(algo_case(kind, len(X), len(X) + 7) * 200)
  call     R = zeros(len(X))
  call     for i to len(X) {
  call         R[i] = gcd(A[i], B[i])
  call     }
  kinds    random dups
  outside  random  "its comment says whole numbers; random gives negatives"

impl gcd_julia  the gcd of the Julia idioms port (numtheory.viv)
  algo     gcd
  module   lib/algo/legacy/gcd_julia
  call     A = floor(X * 200)
  call     B = floor(algo_case(kind, len(X), len(X) + 7) * 200)
  call     R = zeros(len(X))
  call     for i to len(X) {
  call         R[i] = gcd(A[i], B[i])
  call     }
  known    gcd random  "returns a negative divisor when the inputs are negative (gcd(-100, -100) = -100); Julia's gcd is never negative; numtheory.viv calls it only on positive numbers"
  known    oracle random  "the same results, compared with math.gcd"
  kinds    random dups

impl gcd_ta  TheAlgorithms greatest_common_divisor (recursive)
  algo     gcd
  module   lib/algo/legacy/gcd_ta
  call     A = floor(X * 200)
  call     B = floor(algo_case(kind, len(X), len(X) + 7) * 200)
  call     R = zeros(len(X))
  call     for i to len(X) {
  call         R[i] = greatest_common_divisor(A[i], B[i])
  call     }
  kinds    random dups

impl gcd_ta_iterative  TheAlgorithms gcd_by_iterative
  algo     gcd
  module   lib/algo/legacy/gcd_ta
  call     A = floor(X * 200)
  call     B = floor(algo_case(kind, len(X), len(X) + 7) * 200)
  call     R = zeros(len(X))
  call     for i to len(X) {
  call         R[i] = gcd_by_iterative(A[i], B[i])
  call     }
  kinds    random dups

impl ta_permutations  TheAlgorithms all_permutations.py (backtracking), printed with `show`
  algo     next_permutation
  module   lib/algo/legacy/permutations
  call     F = generate_all_permutations(X, false)
  post     factorial-count F X
  oracle   SHOWN itertools-permutations X
  kinds    random dups equal
  sizes    0 1 2 3 4 5

impl ta_topological_sort  TheAlgorithms Kahn topological sort (graphs.viv)
  algo     topological_sort
  module   lib/algo/legacy/topo
  call     K = zeros(len(X))
  call     SRC = []
  call     DST = []
  call     for i to len(X) {
  call         K[i] = len(X) - i
  call     }
  call     for i to len(X) {
  call         for j from i + 1 to len(X) {
  call             if X[i] + X[j] > 0.25 {
  call                 SRC = append(SRC, K[i])
  call                 DST = append(DST, K[j])
  call             }
  call         }
  call     }
  call     R = topological_sort(K, SRC, DST)
  kinds    random
  sizes    0 1 2 3 4 5 7 8 9 15 16 17 31 32 33
