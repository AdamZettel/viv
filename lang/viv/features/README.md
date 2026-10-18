# The feature catalog's sources

`viv features` (lang/viv/src/features.rs) reads these files; the compiler carries
them (build.rs lists the examples), and tests/features.rs checks every one.

| file | what | written |
|---|---|---|
| `<area>.txt` | the language features of one area: syntax, types, arrays, autodiff, correctness, lenses, tools, performance, libraries | by hand |
| `examples/<name>.viv`, `.out` | an entry's example and what it prints | by hand; `.out` recorded |
| `examples/interp/` | examples only the interpreter runs (`engines: interp`) | by hand |
| `examples/modules/` | modules the examples import | by hand |
| `builtins.txt` | one row per builtin form: name, signature, what it does, an example | by hand, checked against the compiler's builtin tables |
| `modules.txt` | lib/ and lang/viv/std, from each module's header | `viv features --write-modules` |

The phrases, the CLI verbs, the lens kinds and the keywords come from the
compiler's own tables (`crate::phrases::TABLE`, `crate::features::VERBS`,
`LENSES`, `KEYWORDS`), and the tests check each table against the code it
describes.

## An entry of `<area>.txt`

    == <id>
    title: <one line>
    what: <what it does; a long value continues on lines indented by two spaces>
    when: <when to use it; what hand-written code it replaces>
    example: <example name>        or   example: none: <why>
    doc: <repo path>[ section N][; <repo path> ...]   (optional: the source that implements it)
    see: <entry id, builtin name, verb or module path>, ...
    keywords: <search words; every keyword of the language is in some entry's>
    usage: <detector>, ...        how `viv features --usage` counts it
    verb: <cli verb>   lens: <lens>   module: <lib path>   engines: interp     (optional)

Detectors: `call:NAME`, `word:NAME`, `use:PATH`, `file:EXT`, `line:PREFIX`, and
`form:NAME` for a form in `crate::features_usage::FORMS`.

## An example

The fewest lines that show the feature, a `show` or `print` of what it did, and
at least one `test` block. It must print the same bytes in the interpreter, as C
and as wasm (`tools/three-engines.sh lang/viv/features/examples`). Its `.out` is
what `viv run -q <name>.viv` prints in this directory. A verb's example adds
`<name>.cmd`: the arguments after `viv`, `{}` for the file, and a second line
`exit N` when the verb ends with status N; its `.out` is what the verb prints.
Lens files (`.intent`, `.debug`, `.memory`, `.arch`) sit beside it.
