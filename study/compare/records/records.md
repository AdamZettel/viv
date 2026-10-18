---
title: Records and dataclasses
order: 1
output: identical
---
- A Viv `record` is a dataclass without methods: a name and typed fields
  (`Num`, `Str`, `Arr`, another record, an array of records). It is a value:
  `q = p` gives `q` its own, so `q.x = 1.0` leaves `p` as it was, where a
  dataclass assignment shares the object. This program builds records and
  never changes their fields, as `@dataclass(frozen=True)` would.
- A record literal names every field, `Vec2 { x: 3.0, y: 4.0 }`. There are no
  positional constructors.
- Methods are plain functions whose first parameter is the record. The call
  `p.plus(q)` works in Viv too: `a.f(b)` is `f(a, b)` for any function.
- `replace(p, y=0.0)` is `Vec2 { y: 0.0, ..p }`, Rust's update syntax.
- An array field is shared, not copied: `earth.pos[1] = 0.25` writes into the
  record's array in both languages. A copy of a Viv record still holds the
  same arrays until ownership's stage 3.
- A dataclass compares with `==`; a Viv record has no comparison (`+ - * /`
  only, over a record of arrays), so the test compares the fields.
