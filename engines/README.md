# engines — Viv programs you feed an input file

An engine is a Viv program with a declared interface:
which files it reads, what it produces, how large an input a browser tab can run, and which
checks it has passed. Each has a page at `viv-lang.org/engines/<name>` where a
reader drops a file and the program runs in the browser.

| engine | reads | runs | held to |
|---|---|---|---|
| `dft-bands.viv` | a crystal: POSCAR, CIF, Quantum ESPRESSO pw.x input, extended XYZ | `dft/pwdft_it.viv`, compiled to one wasm module, or in viv.wasm | dft/'s numpy reference, when the deck is one of dft/'s |
| `hf-energy.viv` | a molecule: XYZ, plus a basis name | `lib/hf/hf.viv` in viv.wasm | PySCF's energy from lib/hfref, when it holds the same calculation |

## The header

```
#! engine dft-bands
#! title Band structure of a crystal, from a POSCAR, a CIF or a pw.x input
#! program dft/pwdft_it
#! input crystal: poscar | cif | qe-pw | xyz+cell
#! output bands, total-energy
#! limit tab: fcc lattices, up to 8 atoms, ...
#! validated si: dft/si_it_check.viv
```

`www/scripts/engine-build.mjs` parses it at build time and fails on an engine
whose `validated` line names a missing file. The lines between
`# ---- input:` and `# ---- end of input` are what the page replaces with the
reader's input; the default there is a complete input, so each engine also runs
as it stands:

```sh
viv run engines/dft-bands.viv     # silicon
viv run engines/hf-energy.viv     # water in 6-31G
```

## Input readers

The input readers, which turn an input file into the engine's input, are
JavaScript, in `www/src/lib/engines/`. They were written before Viv had
strings; a molecule can now also be read in Viv (`molecule(text, basis)`,
lib/hf/mole, which is what `hf-energy.viv`'s default input uses), but the page
keeps its adapter, which also checks the input and estimates the run time:

- `formats/` — POSCAR, CIF (symmetry operations applied), pw.x, extended XYZ;
- `crystal.js` — finds the primitive cell, refuses a lattice that is not fcc,
  and puts the crystal in dft/'s orientation and origin;
- `deck.js` — writes the deck `dft/ref/make_deck.py` would, with the
  pseudopotentials `dft/pseudo/make_pseudo.py` would pick and the settings
  `dft/ref/materials.py` measured (their tables: `dft-tables.json`, from
  `www/scripts/dft-tables.py`);
- `molecule.js` — builds lib/hf's atoms, shells and primitives from PySCF's
  basis library (`hf-bases.json`, from `www/scripts/hf-tables.py`), as
  `tools/hf_reference.py` builds lib/hfref's.

## The compiled DFT engine

`node engines/build-wasm.mjs` compiles `dft-bands.viv` to C, adds a loader
that reads a deck from standard input and overwrites every deck variable by
name, and compiles that with emscripten to `wasm/dft-bands.wasm`. One module
then runs any deck. `--check` says whether the committed module was built from
the current sources (the manifest records the C's SHA-256).

## Tests

```sh
cd www
npm run enginetest           # input readers, decks, packs, refusals: seconds (part of npm run check)
npm run enginerun            # also every example through the engines: about 3 minutes
npm run build && npm run enginebrowsertest   # the pages in Chromium: drop a file, Run, read the numbers
```
