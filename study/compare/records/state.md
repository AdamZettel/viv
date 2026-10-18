---
title: A class with state
order: 3
output: identical
---
- A Python object changes in place: `acct.deposit(25.5)` updates
  `self.balance`. A Viv def does not change a record it is given (a record is
  a value, and a write into a parameter's field is refused), so each
  operation returns a new record and the caller rebinds the name:
  `acct = acct.deposit(25.5)`.
- `__init__` becomes an ordinary function, `open_account`, that builds the
  record with every field named.
- `Account { balance: ..., history: ..., ..a }` is the record `a` with two
  fields replaced.
- `self.history.append(x)` becomes `history = a.history` and then
  `history = append(history, x)`. The append must rebind the name it grows
  (it is refused inside a record literal), and because `a.history` still
  refers to the old array, the append copies it: the old record keeps its old
  history.
- `withdraw` returns `a` unchanged when it refuses; Python's version returns
  `None` and leaves the object as it was.
