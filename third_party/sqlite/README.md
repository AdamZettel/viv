# SQLite amalgamation

- Version: 3.53.4 (`sqlite-amalgamation-3530400.zip`)
- Source: https://sqlite.org/2026/sqlite-amalgamation-3530400.zip, listed on
  https://sqlite.org/download.html
- SHA3-256 of the zip, as that page lists it and as checked on download:
  `628a44cfe82c66aed1ccbbe85a562d2e33ebe64b3288981ed76285612227934e`
- Files kept: `sqlite3.c` and `sqlite3.h`, unmodified.

## License

SQLite is in the public domain. From the head of `sqlite3.c`:

> The author disclaims copyright to this source code. In place of a legal
> notice, here is a blessing: May you do good not evil. May you find
> forgiveness for yourself and forgive others. May you share freely, never
> taking more than you give.

See https://sqlite.org/copyright.html.

## How Viv builds it

The same flags everywhere (`lang/viv/build.rs` for the `viv` binary, and
`viv export --with-sqlite` / `viv run --c` for a program that uses the `db_*`
builtins):

    -DSQLITE_OMIT_LOAD_EXTENSION -DSQLITE_DEFAULT_FOREIGN_KEYS=1

Thread safety is SQLite's default (serialized). JSON functions are built in
since 3.38 and are kept.
