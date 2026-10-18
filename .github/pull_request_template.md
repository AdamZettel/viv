## What this changes

<!-- One paragraph: what the change does and why. Link the design document or issue. -->

## Checklist

- [ ] `tools/gate.sh full` passes on this branch. Summary line:
- [ ] A fixed bug has a line in `bugs/ledger.jsonl` with its test.
- [ ] A new feature has an entry in `lang/viv/features/` with an example.
- [ ] A behavior change has a design document in `design/`, or updates one.
- [ ] Every program touched passes `tools/three-engines.sh` with identical output.
- [ ] Performance numbers in the change cite their result file and say provisional or final.
- [ ] Prose passes `node www/scripts/aispeak.mjs <files>`.
- [ ] Generated files (`www/src/data/`, `KNOWN_LIMITATIONS.md`) are regenerated, not hand-edited.
