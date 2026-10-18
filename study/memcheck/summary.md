| directory | programs | built | ASan errors | UBSan (programs) | ASan run timed out | `leaks` > 0 | arena at exit, unreachable (max) | peak RSS (max) |
|---|---|---|---|---|---|---|---|---|
| dft | 15 | 15 | 0 | 9 | 2 | 0 | 1.13 MB | 133 MB |
| engines | 3 | 3 | 0 | 2 | 0 | 0 | 0.35 MB | 17 MB |
| lib | 120 | 120 | 0 | 22 | 0 | 0 | 1.03 MB | 19 MB |
| pde | 15 | 15 | 0 | 1 | 0 | 0 | 0.18 MB | 3 MB |
| posts | 72 | 72 | 0 | 23 | 0 | 0 | 0.68 MB | 19 MB |
| scf | 30 | 30 | 0 | 27 | 1 | 0 | 0.60 MB | 67 MB |
| study/compare | 37 | 37 | 0 | 0 | 0 | 1 | 0.68 MB | 2 MB |
| study/idioms/syntax-probes | 22 | 22 | 0 | 1 | 0 | 0 | 0.00 MB | 2 MB |
| **all** | 314 | 314 | 0 | 85 | 3 | 1 | | |

UBSan findings, by C function (programs):
- viv_pow: left shift of negative value N (85)

Programs with blocks `leaks` found unreachable:
- study/compare/strings/parse_numbers.viv: 33 blocks, 1680 bytes
