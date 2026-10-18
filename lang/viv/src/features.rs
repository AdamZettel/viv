// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The feature catalog: what Viv can do, for the people and the agents who
//! write it (`viv features`).
//!
//! One list, built from two kinds of source:
//!
//! - **Generated** from the compiler's own tables, so it cannot drift from
//!   them: the builtins (their names, forms and what each does to memory, from
//!   `crate::builtins` and the registries it reads; the one-line description
//!   and the probe program of each are in `features/builtins.txt`, and a test
//!   fails when a builtin the tables know has no row there), the phrases
//!   (`crate::phrases::TABLE`), the CLI verbs ([`VERBS`]), the lenses
//!   ([`LENSES`]), the keywords ([`KEYWORDS`]) and the modules of lib/ and
//!   lang/viv/std (`features/modules.txt`, written from their headers by
//!   `viv features --write-modules`; a test fails when it is stale). The tests
//!   in tests/features.rs check each of these lists against the source that
//!   defines what it lists.
//! - **Written by hand** for what has no table: the language features, by area
//!   (`features/<area>.txt`), each with what it does, when to use it, the
//!   source files that implement it and a minimal example. Every example is a
//!   program in `features/examples/` with a test block and its recorded output;
//!   tests/features.rs runs each in the interpreter, as C and as wasm, and
//!   requires the recorded output from all three.
//!
//! The text formats are described in `features/README.md`.

use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

use crate::debug_lens::json_str;

/// One example file of `features/examples` (listed by build.rs).
pub struct ExampleFile {
    pub name: &'static str,
    /// in `examples/interp/`: the interpreter runs it, the compiled engines do not
    pub interp_only: bool,
    pub viv: &'static str,
    /// what `viv run -q` (or the `.cmd` verb) prints for it
    pub out: Option<&'static str>,
    /// the arguments after `viv` for a verb's output, `{}` for the file; a
    /// second line `exit N` is the exit status it ends with
    pub cmd: Option<&'static str>,
    pub intent: Option<&'static str>,
    pub debug: Option<&'static str>,
    pub memory: Option<&'static str>,
    pub arch: Option<&'static str>,
}

include!(concat!(env!("OUT_DIR"), "/feature_examples.rs"));

/// The areas, in the order the overview shows them: key, heading, what is in it.
pub const AREAS: &[(&str, &str, &str)] = &[
    ("syntax", "Syntax", "the forms the parser reads"),
    ("types", "Types", "the values a program holds"),
    ("arrays", "Arrays", "the NumPy vocabulary and its rules"),
    ("autodiff", "Autodiff", "derivatives of whole defs"),
    ("correctness", "Correctness", "tests, holds, contracts and the checks the compiler makes"),
    ("lenses", "Program knowledge (lenses)", "what a program says about itself beside its code"),
    ("tools", "Tools", "the CLI and the harnesses around it"),
    ("performance", "Performance", "how a program runs fast"),
    ("libraries", "Libraries", "the Viv code a program can import"),
    ("builtins", "Builtins", "generated: features/builtins.txt and the compiler's builtin tables"),
    ("phrases", "Phrases", "generated: crate::phrases::TABLE, by group"),
    ("verbs", "CLI verbs", "generated: crate::features::VERBS"),
    ("lens-kinds", "Lens kinds", "generated: crate::features::LENSES"),
    ("keywords", "Keywords", "generated: the lexer's keywords and the parser's soft keywords"),
    ("modules", "Modules", "generated: the headers of lib/ and lang/viv/std"),
];

/// The hand-written areas.
pub const HAND_AREAS: &[&str] = &["syntax", "types", "arrays", "autodiff", "correctness", "lenses", "tools", "performance", "libraries"];

/// Every verb of the `viv` CLI: its name, its usage and what it does.
/// tests/features.rs checks it against the verbs main.rs dispatches.
pub const VERBS: &[(&str, &str, &str)] = &[
    ("compile", "viv <input.viv | input.py> [-o <output.c|output.wasm>] [--emit tokens|ast|ir]", "compile to C or to wasm, or print the tokens, the AST or the IR"),
    ("run", "viv run [--release] [--quiet] [--json] [--use <prelude.viv>] [--set NAME=value] <input.viv | input.py>\nviv run --fast [--release] [--without contract,libm,blas] <input.viv>", "run in the interpreter; --fast builds the fast native program and runs it"),
    ("check", "viv check [--json] [--use <prelude.viv>] [--set NAME=value] <input.viv | input.py>\nviv check --phrases [--json] [--cert] [--emit out.viv] <input.viv>\nviv check --fast [--ulps N] [--without contract,libm,blas] [--release] [--wasm] [--no-interp] [--json] [--show K] <input.viv>", "the checks a run makes before it runs, and nothing else; --phrases holds each phrase with a closed form to lib/cas's exact value; --fast runs the exact engines and the fast build and compares their output number by number"),
    ("build", "viv build [--fast] [--release] [--single] [--no-cache] [--cache-clear] [--without contract,libm,blas] [--use <prelude.viv>] [--set NAME=value] <input.viv> [-o <executable>]\nviv build --from-split <dir> [--fast] [--release] -o <executable>\nviv build --cc-flags [--fast] [--release]", "a native executable, exact (cc -O3), release or fast, each module compiled once to an object and kept in the cache; --single compiles one C file; --cc-flags prints the C compiler flags of the mode"),
    ("features", "viv features [<name or area> | --search <words> | --json | --usage [<path>...] [--json] | --write-modules]", "this catalog: what Viv can do, and how much of it a corpus uses"),
    ("lint", "viv lint [--count] [--round] <file.viv or directory>...", "integer literals where a size belongs; --round: round(...) an Int would make unnecessary"),
    ("plot", "viv plot [records.out | -] [-o out.svg] [--width N] [--chart K]", "draw the figure records a run printed as SVG"),
    ("export", "viv export <name>.export.toml [-o <dir>] [--reports <dir>]", "a whole program as one C file, a header and a Python wrapper"),
    ("cuda", "viv cuda <input.viv> [-o <output.cu>]", "the CUDA C of the program's GPU kernels and of each `parallel for` that can run on a GPU, with a host launcher per kernel; written, not compiled"),
    ("story", "viv story <file.viv> [--depth <k>] [--expand] [--json] [--engine interp|c] [--from <run output>]", "run in debug mode and tell the run in the intent's words"),
    ("atlas", "viv atlas <file.viv> [-o <file.atlas.html>] [--text | --json] [--no-run] [--no-walk]", "the atlas of a program: its defs by module with their intent, steps and loops, the calls between them and the counts of one run; one HTML file that opens offline, with a walk through the run line by line (recorded by the same command, capped at 4 MB), --text for the terminal, --json for agents, --no-run to skip the run, --no-walk to skip the walk"),
    ("graph", "viv graph <file.viv> [--json | --bundle | --run | --svg [-o out.svg] | --records]", "the program as a graph of defs, steps and loops: the explorer's data"),
    ("src", "viv src <input.viv> <fn>", "one def's source, annotations and all"),
    ("fix", "viv fix --keywords [--write] <file.viv>...", "name every builtin option a file writes by position (`sum(X, axis=-1)`), keeping a file only when its definitions hash as before"),
    ("expand", "viv expand <input.viv>", "the program as the engines read it: its imports spliced in and its names resolved"),
    ("imports", "viv imports [--write] <file.viv>...", "rewrite `#! use` lines as import statements, naming what each file reads from each module"),
    ("ladder", "viv ladder <input.viv>", "run every rung of a ladder and check the ladder's rules"),
    ("lsfunc", "viv lsfunc <input.viv> [--placeholders]", "table of contents: every def and its doc"),
    ("lsvar", "viv lsvar <input.viv> [fn]", "the variable glossary, by def"),
    ("whatis", "viv whatis <input.viv> <name>", "what a name is: glossary, def, record or step"),
    ("doc", "viv doc <input.viv> [fn]", "the annotations as JSON, or one def's doc card"),
    ("deps", "viv deps <input.viv> [--json]", "the call graph"),
    ("canon", "viv canon <input.viv>", "the program in canonical dependency order"),
    ("hash", "viv hash <input.viv>", "each def's node hash and body hash"),
    ("diff", "viv diff <old.viv> <new.viv>", "a structural diff by def: added, removed, changed, renamed"),
    ("db", "viv db query|show|queries|build ...", "the program database through the SQLite inside viv: saved queries and a def's graph or intent; build runs tools/vivdb/vivdb.py"),
    ("claims", "viv claims [--json] [--only <slug>]\nviv claims run [--all | --heavy | --only <slug>] [--write] [--check]", "the claims in claims/ (a statement, its Viv program and its vipyr twin) with their recorded status; `run` runs them through tools/claims/run.py"),
    ("facts", "viv facts --json <input.viv>", "everything the compiler knows about one file, as JSON, for the program database (tools/vivdb)"),
    ("lens", "viv lens init|check|stamp|coverage|needed|move ...", "the intent lens: write, check, stamp and measure it"),
    ("view", "viv view <file.viv> [--lens code|intent|pseudocode|debug|story|memory|all] [--def d] [--words-only] [--json]", "the program through one lens"),
    ("explain", "viv explain <file.viv> [--from <def>] [--depth <k>] [--json]", "the intent of a def and of what it calls, as one explanation"),
    ("help", "viv help [--agent [--brief] | <command> | rules | exit-codes | json] [--json]", "every command; --agent: the working guide for an agent, led by the catalog of validated engines"),
    ("map", "viv map <file.viv> [<def>] [--text | --json] [--summary] [--max-lines N]", "the code map as text: each def under its caller, with its intent, loops and contracts"),
    ("skill", "viv skill install [--project | --user | --plugin] [--force] [--json]", "write the vipyr skill for Claude Code into .claude/skills (claude/ holds the plugin)"),
    ("hook", "viv hook post-edit", "the plugin's PostToolUse hook: check an edited .viv file and show its errors"),
    ("debt", "viv debt [<file.viv|dir>...] [--limit <n>] [--run <program.viv>]", "what the lenses still owe, the most central first"),
    ("arch", "viv arch check|report ...", "the architecture lens: layers, budgets and multi-paths, checked against the code"),
    ("phrases", "viv phrases [--json] [file]", "the phrase table, or what each phrase of a program is"),
    ("cas", "viv cas derivs", "the derivative table as lib/cas/derivs.viv"),
    ("digits", "viv digits [--no-order] [--randoms N] [--no-fma] [--mca mca|rr|pb|off] [--samples N] [--seed N] [--attrib N] [--trace N] [--json] [-o lens.json] [--set NAME=value] <file.viv>", "which digits of the printed numbers are real: reruns with reductions reordered, a*b + c fused and Monte Carlo arithmetic, and the lines that move the rest"),
];

/// The lenses: a view `viv view --lens` shows, or a file beside a module
/// that the tools read. tests/features.rs checks the views against main.rs.
pub const LENSES: &[(&str, &str, &str)] = &[
    ("code", "view", "the source as written"),
    ("intent", "view", "the intent lens: what, how and why of each def and step, from the .intent file or inline"),
    ("pseudocode", "view", "a def as an algorithm box: its intent, and its phrases as words"),
    ("debug", "view", "the debug lens: watches, holds and traces"),
    ("story", "view", "a debug run told in the intent's words"),
    ("memory", "view", "what each def does with the values it is given and makes, inferred"),
    ("all", "view", "every lens of the program at once"),
    (".intent", "file", "the intent lens of a module, beside it"),
    (".debug", "file", "the debug lens of a module: debug blocks the engines append in a debug run"),
    (".memory", "file", "the memory lens: the parallel loops pinned as disjoint"),
    (".arch", "file", "the architecture lens: layers, budgets and multi-paths"),
];

/// Every keyword the lexer reserves and every soft keyword the parser reads
/// as a word. tests/features.rs checks this list against lexer.rs and
/// parser.rs, and that each is in the `keywords:` of some hand-written entry.
pub const KEYWORDS: &[&str] = &[
    // the lexer's
    "def", "return", "if", "else", "while", "for", "to", "break", "continue", "narrate", "show", "test", "rung", "parallel", "fn", "record", "and",
    "or", "not", "true", "false",
    // the parser's soft keywords
    "enum", "match", "in", "from", "by", "reversed", "step", "debug", "placeholder", "phrase", "take", "using", "starting", "with", "watch", "hold",
    "trace", "within", "per", "iteration", "at", "key", "sep", "start",
    // `kernel def` (crate::cuda)
    "kernel",
    // `schedule f(blocks = ..., threads = ...)` (crate::cuda)
    "schedule",
    // `on device a, b { ... }` (crate::cuda)
    "on", "device",
    // imports (crate::imports): `import a.b as c`, `from a import b`, `private def`
    "import", "as", "private",
    // type names the parser reads as words (in `[Str]` and `[Int]`); they are
    // not reserved, the other type names (Num, Arr, ...) are plain identifiers
    "Int", "Str",
    // the match wildcard: `_ => ...` and `B(x, _)`
    "_",
];

/// A way to count a feature's uses in source text (`viv features --usage`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Detector {
    /// a call `name(...)`
    Call(String),
    /// the identifier anywhere in code
    Word(String),
    /// a `#! use` of the module or one under it
    Use(String),
    /// a companion file `<module>.<ext>`
    File(String),
    /// a line that starts (after spaces) with the text
    Line(String),
    /// a syntax form `crate::features_usage` knows
    Form(String),
}

impl Detector {
    pub fn parse(s: &str) -> Result<Detector, String> {
        let (kind, arg) = s.split_once(':').ok_or_else(|| format!("`{s}`: a detector is kind:argument"))?;
        let arg = arg.trim().to_string();
        if arg.is_empty() {
            return Err(format!("`{s}`: no argument"));
        }
        Ok(match kind.trim() {
            "call" => Detector::Call(arg),
            "word" => Detector::Word(arg),
            "use" => Detector::Use(arg),
            "file" => Detector::File(arg.trim_start_matches('.').to_string()),
            "line" => Detector::Line(arg),
            "form" => {
                if !crate::features_usage::FORMS.contains(&arg.as_str()) {
                    return Err(format!("`{s}`: no form `{arg}` (crate::features_usage::FORMS)"));
                }
                Detector::Form(arg)
            }
            k => return Err(format!("`{s}`: no detector kind `{k}` (call, word, use, file, line, form)")),
        })
    }

    pub fn text(&self) -> String {
        match self {
            Detector::Call(a) => format!("call:{a}"),
            Detector::Word(a) => format!("word:{a}"),
            Detector::Use(a) => format!("use:{a}"),
            Detector::File(a) => format!("file:{a}"),
            Detector::Line(a) => format!("line:{a}"),
            Detector::Form(a) => format!("form:{a}"),
        }
    }
}

/// One form of a generated entry: a signature, a pattern, a usage line.
#[derive(Clone, Debug)]
pub struct Form {
    pub text: String,
    pub note: String,
}

/// An example and what it prints.
#[derive(Clone, Debug)]
pub struct Example {
    pub name: String,
    pub code: String,
    /// recorded (and checked by tests/features.rs), or computed now
    pub output: Option<String>,
    /// for a verb's output: the command, `viv ...`
    pub cmd: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub id: String,
    pub area: String,
    /// the builtin group or phrase group, else empty
    pub group: String,
    pub title: String,
    pub what: String,
    pub when: String,
    pub forms: Vec<Form>,
    pub examples: Vec<Example>,
    /// why there is no example, when there is none
    pub no_example: String,
    /// the source files (or reference sections) that implement or document it
    pub doc: Vec<String>,
    pub see: Vec<String>,
    pub keywords: Vec<String>,
    pub usage: Vec<Detector>,
    pub verb: Option<String>,
    pub lens: Option<String>,
    pub module: Option<String>,
    pub interp_only: bool,
    /// where it comes from: `features/<area>.txt` or the generating table
    pub source: String,
    /// for a builtin: what it does to memory (crate::builtins)
    pub memory: String,
}

impl Entry {
    fn new(id: &str, area: &str, source: &str) -> Entry {
        Entry {
            id: id.to_string(),
            area: area.to_string(),
            group: String::new(),
            title: String::new(),
            what: String::new(),
            when: String::new(),
            forms: Vec::new(),
            examples: Vec::new(),
            no_example: String::new(),
            doc: Vec::new(),
            see: Vec::new(),
            keywords: Vec::new(),
            usage: Vec::new(),
            verb: None,
            lens: None,
            module: None,
            interp_only: false,
            source: source.to_string(),
            memory: String::new(),
        }
    }

    pub fn hand(&self) -> bool {
        HAND_AREAS.contains(&self.area.as_str())
    }

    /// Every word a search reads.
    fn haystack(&self) -> String {
        let mut s = format!("{} {} {} {} {} {} {}", self.id, self.title, self.what, self.when, self.group, self.keywords.join(" "), self.area);
        for f in &self.forms {
            s.push(' ');
            s.push_str(&f.text);
            s.push(' ');
            s.push_str(&f.note);
        }
        s.to_lowercase()
    }
}

// ---- the text format ---------------------------------------------------------

/// One `== id` block of a catalog file: its fields in order, with the line it starts on.
pub struct Block {
    pub id: String,
    pub line: usize,
    pub fields: Vec<(String, String)>,
}

impl Block {
    pub fn get(&self, k: &str) -> Option<&str> {
        self.fields.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
    }
}

/// The fields an entry of a hand-written area may have.
pub const FIELDS: &[&str] = &["title", "what", "when", "example", "doc", "see", "keywords", "usage", "verb", "lens", "module", "engines"];

/// Parse a catalog file: `== id` opens a block, `key: value` lines are its
/// fields, a line indented by two spaces continues the field before it, and a
/// blank line or a `#` line at column 0 is skipped.
pub fn parse_blocks(text: &str) -> Result<Vec<Block>, Vec<String>> {
    let mut out: Vec<Block> = Vec::new();
    let mut errs = Vec::new();
    let mut open = false;
    for (k, raw) in text.lines().enumerate() {
        let n = k + 1;
        if let Some(id) = raw.strip_prefix("== ") {
            out.push(Block { id: id.trim().to_string(), line: n, fields: Vec::new() });
            open = false;
            continue;
        }
        if raw.trim().is_empty() {
            open = false;
            continue;
        }
        if raw.starts_with('#') {
            continue;
        }
        if let (Some(rest), true) = (raw.strip_prefix("  "), open) {
            if let Some((_, v)) = out.last_mut().and_then(|b| b.fields.last_mut()) {
                v.push(' ');
                v.push_str(rest.trim());
            }
            continue;
        }
        let Some(b) = out.last_mut() else {
            errs.push(format!("line {n}: a field before the first `== id`"));
            continue;
        };
        match raw.split_once(':') {
            Some((key, v)) if !key.is_empty() && key.chars().all(|c| c.is_ascii_lowercase()) => {
                if b.fields.iter().any(|(f, _)| f == key) {
                    errs.push(format!("line {n}: `{key}` twice in `{}`", b.id));
                }
                b.fields.push((key.to_string(), v.trim().to_string()));
                open = true;
            }
            _ => errs.push(format!("line {n}: not a `key: value` line: {raw}")),
        }
    }
    if errs.is_empty() { Ok(out) } else { Err(errs) }
}

fn list(v: Option<&str>, sep: char) -> Vec<String> {
    v.map(|s| s.split(sep).map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()).unwrap_or_default()
}

/// The example files by name.
fn example_file(name: &str) -> Option<&'static ExampleFile> {
    EXAMPLE_FILES.iter().find(|f| f.name == name)
}

fn example_of(f: &ExampleFile) -> Example {
    let (cmd, _) = cmd_of(f);
    Example { name: f.name.to_string(), code: f.viv.to_string(), output: f.out.map(str::to_string), cmd: cmd.map(|args| format!("viv {}", args.join(" "))) }
}

/// The arguments after `viv` of an example's `.cmd`, with `{}` as its file
/// name, and the exit status it ends with.
pub fn cmd_of(f: &ExampleFile) -> (Option<Vec<String>>, i32) {
    let Some(text) = f.cmd else { return (None, 0) };
    let mut lines = text.lines();
    let args = lines.next().unwrap_or("").split_whitespace().map(|a| a.replace("{}", &format!("{}.viv", f.name))).collect();
    let exit = lines.find_map(|l| l.trim().strip_prefix("exit ").and_then(|n| n.trim().parse().ok())).unwrap_or(0);
    (Some(args), exit)
}

/// The entries of one hand-written area file.
fn hand_entries(area: &str, text: &str, errs: &mut Vec<String>) -> Vec<Entry> {
    let blocks = match parse_blocks(text) {
        Ok(b) => b,
        Err(e) => {
            errs.extend(e.into_iter().map(|m| format!("features/{area}.txt: {m}")));
            return Vec::new();
        }
    };
    let mut out = Vec::new();
    for b in blocks {
        let at = format!("features/{area}.txt:{} `{}`", b.line, b.id);
        for (k, _) in &b.fields {
            if !FIELDS.contains(&k.as_str()) {
                errs.push(format!("{at}: no field `{k}` (fields: {})", FIELDS.join(", ")));
            }
        }
        let mut e = Entry::new(&b.id, area, &format!("features/{area}.txt"));
        e.title = b.get("title").unwrap_or("").to_string();
        e.what = b.get("what").unwrap_or("").to_string();
        e.when = b.get("when").unwrap_or("").to_string();
        e.doc = list(b.get("doc"), ';');
        e.see = list(b.get("see"), ',');
        e.keywords = b.get("keywords").map(|k| k.split_whitespace().map(str::to_string).collect()).unwrap_or_default();
        e.verb = b.get("verb").map(str::to_string);
        e.lens = b.get("lens").map(str::to_string);
        e.module = b.get("module").map(str::to_string);
        e.interp_only = b.get("engines") == Some("interp");
        for d in list(b.get("usage"), ',') {
            match Detector::parse(&d) {
                Ok(d) => e.usage.push(d),
                Err(m) => errs.push(format!("{at}: {m}")),
            }
        }
        match b.get("example") {
            Some(x) if x.starts_with("none") => e.no_example = x.trim_start_matches("none").trim_start_matches(':').trim().to_string(),
            Some(name) => match example_file(name) {
                Some(f) => {
                    if f.interp_only != e.interp_only {
                        errs.push(format!("{at}: example `{name}` is in examples/{} and the entry says {}", if f.interp_only { "interp/" } else { "" }, if e.interp_only { "engines: interp" } else { "nothing of engines" }));
                    }
                    e.examples.push(example_of(f));
                }
                None => errs.push(format!("{at}: no example features/examples/{name}.viv")),
            },
            None => {}
        }
        for k in ["title", "what", "when", "example"] {
            if b.get(k).map_or(true, |v| v.is_empty()) {
                errs.push(format!("{at}: no `{k}`"));
            }
        }
        out.push(e);
    }
    out
}

/// Undo the escapes of a builtins.txt example: `\n` is a line break, `\\` a backslash.
fn unescape(s: &str) -> String {
    let mut out = String::new();
    let mut cs = s.chars();
    while let Some(c) = cs.next() {
        if c == '\\' {
            match cs.next() {
                Some('n') => out.push('\n'),
                Some('\\') => out.push('\\'),
                Some(o) => {
                    out.push('\\');
                    out.push(o);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// One row of features/builtins.txt.
pub struct BuiltinRow {
    pub group: String,
    pub note: String,
    pub name: String,
    pub sig: String,
    pub blurb: String,
    pub probe: String,
}

/// The rows of features/builtins.txt, in order.
pub fn builtin_rows() -> Vec<BuiltinRow> {
    let text = catalog_text("builtins");
    let (mut group, mut note) = (String::new(), String::new());
    let mut out = Vec::new();
    for line in text.lines() {
        if let Some(g) = line.strip_prefix("## ") {
            group = g.trim().to_string();
            note.clear();
        } else if let Some(n) = line.strip_prefix("note: ") {
            note = n.trim().to_string();
        } else if line.contains('\t') {
            let f: Vec<&str> = line.split('\t').collect();
            out.push(BuiltinRow {
                group: group.clone(),
                note: note.clone(),
                name: f[0].to_string(),
                sig: f.get(1).unwrap_or(&"").to_string(),
                blurb: f.get(2).unwrap_or(&"").to_string(),
                probe: unescape(f.get(3).unwrap_or(&"")),
            });
        }
    }
    out
}

/// The names a signature calls: `max(a, s) · max(a, b)` gives `max`.
pub fn sig_names(sig: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let cs: Vec<char> = sig.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        if cs[i].is_alphabetic() || cs[i] == '_' {
            let s = i;
            while i < cs.len() && (cs[i].is_alphanumeric() || cs[i] == '_') {
                i += 1;
            }
            let word: String = cs[s..i].iter().collect();
            let mut j = i;
            while j < cs.len() && cs[j] == ' ' {
                j += 1;
            }
            let after_dot = s > 0 && cs[s - 1] == '.';
            if j < cs.len() && cs[j] == '(' && !after_dot && !out.contains(&word) {
                out.push(word);
            }
        } else {
            i += 1;
        }
    }
    out
}

fn memory_of(name: &str) -> String {
    use crate::builtins::Kind;
    match crate::builtins::kind(name) {
        Some(Kind::Writes) => {
            let r = crate::builtins::writes(name).or_else(|| crate::interp::arrays::SIGS.iter().find(|s| s.name == name).and_then(|s| s.writes).map(|k| k..=k));
            match r {
                Some(r) if r.start() == r.end() => format!("writes its argument {} in place, when a call gives it", r.start() + 1),
                Some(r) => format!("writes its arguments {} to {} in place, when a call gives them", r.start() + 1, r.end() + 1),
                None => "writes an argument in place".to_string(),
            }
        }
        Some(Kind::Pure) => "pure: a value from its arguments, nothing changed (grad may recompute it; a hold may call it)".to_string(),
        Some(Kind::Effect) => "an effect: output, input, the random generator, the test counters or a call of a function value".to_string(),
        Some(Kind::Reads) => "reads its arguments, writes none".to_string(),
        None => String::new(),
    }
}

fn builtin_entries() -> Vec<Entry> {
    let mut by_name: Vec<Entry> = Vec::new();
    for r in builtin_rows() {
        let id = r.name.clone();
        let i = match by_name.iter().position(|e| e.id == id) {
            Some(i) => i,
            None => {
                let mut e = Entry::new(&id, "builtins", "features/builtins.txt and crate::builtins");
                e.group = r.group.clone();
                e.title = r.sig.clone();
                e.what = r.blurb.clone();
                e.when = r.note.clone();
                e.doc = vec!["www/src/pages/reference.astro (Builtins)".to_string()];
                by_name.push(e);
                by_name.len() - 1
            }
        };
        let e = &mut by_name[i];
        if e.what.is_empty() {
            e.what = r.blurb.clone();
        }
        e.forms.push(Form { text: r.sig.clone(), note: r.blurb.clone() });
        for n in sig_names(&r.sig) {
            let d = Detector::Call(n.clone());
            if !e.usage.contains(&d) {
                e.usage.push(d);
            }
            if e.memory.is_empty() {
                e.memory = memory_of(&n);
            }
            if !e.keywords.contains(&n) {
                e.keywords.push(n);
            }
        }
        if !r.probe.is_empty() {
            let k = e.examples.len();
            e.examples.push(Example { name: if k == 0 { e.id.clone() } else { format!("{} ({})", e.id, k + 1) }, code: r.probe.clone(), output: None, cmd: None });
        }
    }
    by_name
}

fn group_slug(g: &str) -> String {
    let mut s = String::new();
    for c in g.chars() {
        if c.is_ascii_alphanumeric() {
            s.push(c.to_ascii_lowercase());
        } else if !s.ends_with('-') {
            s.push('-');
        }
    }
    s.trim_matches('-').to_string()
}

/// The phrase examples: tests/phrases/phrases.viv, one test per phrase, named by the phrase.
const PHRASE_TESTS: &str = include_str!("../tests/phrases/phrases.viv");

fn phrase_entries() -> Vec<Entry> {
    let mut out: Vec<Entry> = Vec::new();
    let tests: Vec<&str> = PHRASE_TESTS.lines().filter_map(|l| l.strip_prefix("test \"").and_then(|r| r.strip_suffix("\" {"))).collect();
    for (k, p) in crate::phrases::TABLE.iter().enumerate() {
        let id = format!("phrases-{}", group_slug(p.group));
        let i = match out.iter().position(|e| e.id == id) {
            Some(i) => i,
            None => {
                let mut e = Entry::new(&id, "phrases", "crate::phrases::TABLE");
                e.group = p.group.to_string();
                e.title = format!("Phrases: {}", p.group.to_lowercase());
                e.what = "Math written as words; the parser reads each phrase as the call beside it, so every engine runs only calls it already runs.".to_string();
                e.when = "When the line reads better as the sentence a person would say; the call is the same code.".to_string();
                e.doc = vec!["lang/viv/src/phrases.rs".to_string()];
                e.see = vec!["phrases".to_string(), "user-phrases".to_string()];
                e.usage.push(Detector::Form(format!("phrase-group:{}", p.group)));
                out.push(e);
                out.len() - 1
            }
        };
        let written = crate::phrases::written(p);
        out[i].forms.push(Form { text: written.clone(), note: p.lowers.to_string() });
        // the shortest test whose name this pattern reads, as phrases-build.mjs picks it
        let ex = tests.iter().filter(|t| phrase_matches(p, t)).min_by_key(|t| t.len());
        if let Some(t) = ex {
            if out[i].examples.len() < 3 {
                out[i].examples.push(Example { name: format!("phrase {k}"), code: t.to_string(), output: None, cmd: None });
            }
        }
    }
    out
}

/// Whether a phrase as written (a test's name) reads pattern `p`, the way
/// www/scripts/phrases-build.mjs matches them: literal words in order, any
/// text for a hole.
fn phrase_matches(p: &crate::phrases::Pattern, text: &str) -> bool {
    let words: Vec<&str> = p.words.split_whitespace().collect();
    let mut rest = text.trim();
    if crate::phrases::bare_verb(p).is_none() {
        match rest.strip_prefix("take ") {
            Some(r) => rest = r,
            None => return false,
        }
    }
    fn go(words: &[&str], rest: &str) -> bool {
        let Some((w, more)) = words.split_first() else { return rest.is_empty() };
        if w.starts_with('{') {
            // a hole takes one or more words
            let toks: Vec<&str> = rest.split(' ').collect();
            for n in 1..=toks.len() {
                let tail = toks[n..].join(" ");
                if go(more, &tail) {
                    return true;
                }
            }
            return false;
        }
        let alts: Vec<String> = w.split('|').map(|a| a.replace('-', " - ")).collect();
        for a in alts {
            let a = a.replace(" - ", "-");
            if let Some(r) = rest.strip_prefix(a.as_str()) {
                if r.is_empty() || r.starts_with(' ') {
                    if go(more, r.trim_start()) {
                        return true;
                    }
                }
            }
        }
        false
    }
    go(&words, rest)
}

fn verb_entries(hand: &[Entry]) -> Vec<Entry> {
    VERBS
        .iter()
        .map(|(name, usage, what)| {
            let mut e = Entry::new(&format!("verb-{name}"), "verbs", "crate::features::VERBS");
            e.title = format!("viv {name}");
            e.what = what.to_string();
            e.forms = usage.lines().map(|u| Form { text: u.to_string(), note: String::new() }).collect();
            e.doc = vec!["lang/viv/src/main.rs".to_string()];
            e.keywords = vec![name.to_string()];
            if let Some(h) = hand.iter().find(|h| h.verb.as_deref() == Some(name)) {
                e.see.push(h.id.clone());
                e.when = format!("See `{}` for when to use it and an example.", h.id);
            }
            e
        })
        .collect()
}

fn lens_entries(hand: &[Entry]) -> Vec<Entry> {
    LENSES
        .iter()
        .map(|(name, kind, what)| {
            let id = if kind == &"file" { format!("lens-file{}", name.replace('.', "-")) } else { format!("lens-{name}") };
            let mut e = Entry::new(&id, "lens-kinds", "crate::features::LENSES");
            e.title = if kind == &"file" { format!("{name} file") } else { format!("viv view --lens {name}") };
            e.what = what.to_string();
            e.doc = vec!["lang/viv/src/lens.rs".to_string()];
            e.keywords = vec![name.trim_start_matches('.').to_string(), "lens".to_string()];
            if kind == &"file" {
                e.usage.push(Detector::File(name.trim_start_matches('.').to_string()));
            }
            let key = name.trim_start_matches('.');
            if let Some(h) = hand.iter().find(|h| h.lens.as_deref() == Some(key) || h.lens.as_deref() == Some(name)) {
                e.see.push(h.id.clone());
                e.when = format!("See `{}`.", h.id);
            }
            e
        })
        .collect()
}

fn keyword_entries(hand: &[Entry]) -> Vec<Entry> {
    KEYWORDS
        .iter()
        .map(|k| {
            let mut e = Entry::new(&format!("keyword-{k}"), "keywords", "crate::features::KEYWORDS");
            e.title = format!("`{k}`");
            let docs: Vec<String> = hand.iter().filter(|h| h.keywords.iter().any(|w| w == k)).map(|h| h.id.clone()).collect();
            e.what = if docs.is_empty() { "No entry documents it yet.".to_string() } else { format!("Documented by: {}.", docs.join(", ")) };
            e.see = docs;
            e.keywords = vec![k.to_string()];
            e.usage.push(Detector::Word(k.to_string()));
            e
        })
        .collect()
}

fn module_entries(hand: &[Entry], errs: &mut Vec<String>) -> Vec<Entry> {
    let blocks = match parse_blocks(catalog_text("modules")) {
        Ok(b) => b,
        Err(e) => {
            errs.extend(e.into_iter().map(|m| format!("features/modules.txt: {m}")));
            return Vec::new();
        }
    };
    blocks
        .into_iter()
        .map(|b| {
            let mut e = Entry::new(&b.id, "modules", "features/modules.txt (viv features --write-modules)");
            e.title = b.get("title").unwrap_or("").to_string();
            e.what = b.get("summary").unwrap_or("").to_string();
            let defs = b.get("defs").unwrap_or("0");
            let names = b.get("names").unwrap_or("");
            e.forms.push(Form { text: format!("import {}", crate::imports::dotted_of(&b.id)), note: format!("{defs} defs: {names}") });
            e.doc = vec![format!("{}.viv", b.id)];
            e.usage.push(Detector::Use(b.id.clone()));
            e.keywords = names.split(',').map(|n| n.trim().to_string()).filter(|n| !n.is_empty()).collect();
            if let Some(h) = hand.iter().find(|h| h.module.as_deref() == Some(b.id.as_str())) {
                e.see.push(h.id.clone());
            }
            e.group = b.id.rsplit_once('/').map(|(d, _)| d.to_string()).unwrap_or_default();
            e
        })
        .collect()
}

/// The text of a catalog file (`syntax`, ..., `builtins`, `modules`).
pub fn catalog_text(name: &str) -> &'static str {
    CATALOG_FILES.iter().find(|(n, _)| *n == name).map(|(_, t)| *t).unwrap_or("")
}

/// The catalog and the problems found building it (none in a good build:
/// tests/features.rs requires that).
pub struct Catalog {
    pub entries: Vec<Entry>,
    pub errors: Vec<String>,
}

/// Build the catalog from its sources.
pub fn build() -> Catalog {
    let mut errors = Vec::new();
    let mut entries = Vec::new();
    for area in HAND_AREAS {
        entries.extend(hand_entries(area, catalog_text(area), &mut errors));
    }
    let hand: Vec<Entry> = entries.clone();
    // a builtin whose name a hand-written entry has as its id goes by `<name> (builtin)`
    entries.extend(builtin_entries().into_iter().map(|mut b| {
        if hand.iter().any(|h| h.id == b.id) {
            b.id = format!("{} (builtin)", b.id);
        }
        b
    }));
    entries.extend(phrase_entries());
    entries.extend(verb_entries(&hand));
    entries.extend(lens_entries(&hand));
    entries.extend(keyword_entries(&hand));
    entries.extend(module_entries(&hand, &mut errors));
    let mut seen: HashMap<&str, usize> = HashMap::new();
    for (k, e) in entries.iter().enumerate() {
        if let Some(j) = seen.insert(e.id.as_str(), k) {
            errors.push(format!("`{}` is the id of two entries ({} and {})", e.id, entries[j].source, e.source));
        }
    }
    // every `see` names an entry, or a builtin or verb by its call name
    let ids: std::collections::HashSet<&str> = entries.iter().map(|e| e.id.as_str()).collect();
    for e in entries.iter().filter(|e| e.hand()) {
        for s in &e.see {
            if !ids.contains(s.as_str()) && !ids.contains(format!("verb-{s}").as_str()) && resolve_name(&entries, s).is_empty() {
                errors.push(format!("{} `{}`: `see: {s}` names no entry", e.source, e.id));
            }
        }
    }
    Catalog { entries, errors }
}

/// The catalog, built once.
pub fn catalog() -> &'static Catalog {
    static C: OnceLock<Catalog> = OnceLock::new();
    C.get_or_init(build)
}

/// The entries a name finds: its id, a builtin by a name its forms call, a
/// verb by its name, a keyword, a module by its path.
fn resolve_name<'a>(entries: &'a [Entry], q: &str) -> Vec<&'a Entry> {
    let q = q.trim();
    let mut out: Vec<&Entry> = entries.iter().filter(|e| e.id == q).collect();
    let exact_hand = out.iter().any(|e| e.hand());
    for e in entries {
        if out.iter().any(|o| o.id == e.id) {
            continue;
        }
        // an exact hand-written entry, and the builtin of that name, are what it finds
        if !out.is_empty() && !(exact_hand && e.area == "builtins") {
            continue;
        }
        let hit = match e.area.as_str() {
            "builtins" => e.usage.iter().any(|d| *d == Detector::Call(q.to_string())) || e.id.split_whitespace().next() == Some(q),
            "verbs" => e.id == format!("verb-{q}") || e.title == q || e.title == format!("viv {q}"),
            "keywords" => e.id == format!("keyword-{q}"),
            "modules" => e.id.trim_start_matches("lib/") == q.trim_start_matches("lib/"),
            _ => e.verb.as_deref() == Some(q) || e.module.as_deref() == Some(q),
        };
        if hit {
            out.push(e);
        }
    }
    out
}

// ---- text views ------------------------------------------------------------

fn area_heading(area: &str) -> &str {
    AREAS.iter().find(|(k, _, _)| *k == area).map(|(_, h, _)| *h).unwrap_or(area)
}

/// Wrap `text` to `width` columns, each line starting with `indent`.
fn wrap(text: &str, indent: &str, width: usize) -> String {
    let mut out = String::new();
    let mut line = String::new();
    for w in text.split_whitespace() {
        if !line.is_empty() && indent.len() + line.chars().count() + 1 + w.chars().count() > width {
            out.push_str(indent);
            out.push_str(&line);
            out.push('\n');
            line.clear();
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(w);
    }
    if !line.is_empty() {
        out.push_str(indent);
        out.push_str(&line);
        out.push('\n');
    }
    out
}

/// `viv features`: every area, its entries' ids, and how to read more.
pub fn overview() -> String {
    let c = catalog();
    let mut out = String::from("Viv's features, by area. `viv features <name or area>` for one; `--search <words>`; `--usage [paths]`.\n");
    for (key, heading, what) in AREAS {
        let es: Vec<&Entry> = c.entries.iter().filter(|e| e.area == *key).collect();
        if es.is_empty() {
            continue;
        }
        out.push_str(&format!("\n{heading} ({}): {what}\n", es.len()));
        match *key {
            "builtins" | "modules" => {
                let mut groups: Vec<(String, usize)> = Vec::new();
                for e in &es {
                    match groups.iter_mut().find(|(g, _)| *g == e.group) {
                        Some(g) => g.1 += 1,
                        None => groups.push((e.group.clone(), 1)),
                    }
                }
                let line = groups.iter().map(|(g, n)| format!("{} {n}", if g.is_empty() { "(top)" } else { g })).collect::<Vec<_>>().join(" · ");
                out.push_str(&wrap(&line, "  ", 100));
            }
            "keywords" => {
                let line = es.iter().map(|e| e.id.trim_start_matches("keyword-").to_string()).collect::<Vec<_>>().join(" ");
                out.push_str(&wrap(&line, "  ", 100));
            }
            "verbs" | "lens-kinds" => {
                let line = es.iter().map(|e| e.id.trim_start_matches("verb-").trim_start_matches("lens-").to_string()).collect::<Vec<_>>().join(" ");
                out.push_str(&wrap(&line, "  ", 100));
            }
            _ => {
                let w = es.iter().map(|e| e.id.chars().count()).max().unwrap_or(0).min(30);
                for e in es {
                    out.push_str(&format!("  {:<w$}  {}\n", e.id, e.title, w = w));
                }
            }
        }
    }
    let hand = c.entries.iter().filter(|e| e.hand()).count();
    out.push_str(&format!("\n{} entries: {hand} written by hand, {} generated from the compiler's tables.\n", c.entries.len(), c.entries.len() - hand));
    out
}

/// One area's entries, each with its first sentence.
fn area_view(area: &str) -> String {
    let c = catalog();
    let es: Vec<&Entry> = c.entries.iter().filter(|e| e.area == area).collect();
    let (_, heading, what) = AREAS.iter().find(|(k, _, _)| *k == area).copied().unwrap_or((area, area, ""));
    let mut out = format!("{heading}: {what} ({} entries)\n\n", es.len());
    let w = es.iter().map(|e| e.id.chars().count()).max().unwrap_or(0).min(34);
    for e in es {
        let line = if e.area == "builtins" || e.area == "verbs" || e.area == "phrases" { e.title.clone() } else { first_sentence(&e.what) };
        out.push_str(&format!("  {:<w$}  {}\n", e.id, line, w = w));
    }
    out
}

fn first_sentence(s: &str) -> String {
    let mut out = String::new();
    let cs: Vec<char> = s.chars().collect();
    for (i, c) in cs.iter().enumerate() {
        out.push(*c);
        if *c == '.' && cs.get(i + 1).map_or(true, |n| *n == ' ') && out.len() > 20 {
            break;
        }
    }
    out
}

/// The output of an example: recorded, or (for a builtin's or a phrase's
/// example) what the interpreter prints for it now.
pub fn example_output(e: &Example) -> String {
    if let Some(o) = &e.output {
        return o.clone();
    }
    let text = if e.name.starts_with("phrase ") { phrase_program(&e.code) } else { e.code.clone() };
    // lib/cas, which the compiler carries, is the one module these examples use
    let cas_import = crate::imports::scan(&text).map(|ss| ss.iter().any(|s| s.items.iter().any(|i| i.path().first().is_some_and(|p| p == "cas")))).unwrap_or(false);
    let text = if text.contains("#! use lib/cas/") || cas_import {
        crate::expand_uses(&text, &mut |name: &str| crate::cas_check::lib_module(name).map(str::to_string).ok_or_else(|| name.to_string())).unwrap_or_else(|_| text.clone())
    } else {
        text
    };
    match crate::interp::run(&text) {
        Ok(out) => out.lines().filter(|l| !l.starts_with("__")).map(|l| format!("{l}\n")).collect(),
        Err(err) => format!("error: {err}\n"),
    }
}

/// The test of tests/phrases/phrases.viv named `name`, as a program of its own.
fn phrase_program(name: &str) -> String {
    let head = format!("test \"{name}\" {{");
    let mut out = String::new();
    let mut lines = PHRASE_TESTS.lines();
    // the file's top-level lines before its first test (shared data)
    for l in PHRASE_TESTS.lines() {
        if l.starts_with("test \"") {
            break;
        }
        out.push_str(l);
        out.push('\n');
    }
    for l in lines.by_ref() {
        if l == head {
            out.push_str(l);
            out.push('\n');
            break;
        }
    }
    for l in lines {
        out.push_str(l);
        out.push('\n');
        if l == "}" {
            break;
        }
    }
    out
}

/// The example code as the detail view shows it: a phrase's test block.
fn example_code(e: &Example) -> String {
    if e.name.starts_with("phrase ") {
        let p = phrase_program(&e.code);
        let start = p.find(&format!("test \"{}\"", e.code)).unwrap_or(0);
        return p[start..].to_string();
    }
    e.code.clone()
}

fn indent(text: &str, by: &str) -> String {
    text.lines().map(|l| if l.is_empty() { "\n".to_string() } else { format!("{by}{l}\n") }).collect()
}

/// The detail of one entry.
pub fn detail(e: &Entry) -> String {
    let mut out = format!("{}  [{}{}]\n", e.id, area_heading(&e.area), if e.group.is_empty() { String::new() } else { format!(" / {}", e.group) });
    if !e.title.is_empty() && e.title != e.id {
        out.push_str(&format!("  {}\n", e.title));
    }
    out.push('\n');
    if !e.what.is_empty() {
        out.push_str("What it does\n");
        out.push_str(&wrap(&e.what, "  ", 92));
    }
    if !e.when.is_empty() {
        out.push_str("When to use it\n");
        out.push_str(&wrap(&e.when, "  ", 92));
    }
    if !e.forms.is_empty() && !(e.forms.len() == 1 && e.forms[0].text == e.title && e.forms[0].note == e.what) {
        out.push_str("Forms\n");
        for f in &e.forms {
            if f.note.is_empty() || (e.area == "builtins" && f.note == e.what) {
                out.push_str(&format!("  {}\n", f.text));
            } else if e.area == "phrases" {
                out.push_str(&format!("  {}\n      = {}\n", f.text, f.note));
            } else {
                out.push_str(&format!("  {}\n", f.text));
                out.push_str(&wrap(&f.note, "      ", 92));
            }
        }
    }
    if !e.memory.is_empty() {
        out.push_str(&format!("Memory\n  {}\n", e.memory));
    }
    if e.interp_only {
        out.push_str("Engines\n  the interpreter only: the compiled engines refuse it\n");
    }
    for ex in &e.examples {
        out.push_str(&format!("Example{}\n", if e.examples.len() > 1 { format!(" ({})", ex.name) } else { String::new() }));
        out.push_str(&indent(&example_code(ex), "  "));
        match &ex.cmd {
            Some(cmd) => out.push_str(&format!("Output of `{cmd}`\n")),
            None => out.push_str(&format!("Output{}\n", if ex.output.is_some() { "" } else { " (the interpreter, now)" })),
        }
        out.push_str(&indent(&example_output(ex), "  "));
    }
    if e.examples.is_empty() && !e.no_example.is_empty() {
        out.push_str(&format!("Example\n  none: {}\n", e.no_example));
    }
    if !e.doc.is_empty() {
        out.push_str(&format!("Source\n  {}\n", e.doc.join("; ")));
    }
    if !e.see.is_empty() {
        out.push_str(&format!("Related\n  {}\n", e.see.join(", ")));
    }
    if !e.usage.is_empty() {
        out.push_str(&format!("Counted by `viv features --usage` as\n  {}\n", e.usage.iter().map(|d| d.text()).collect::<Vec<_>>().join(", ")));
    }
    out.push_str(&format!("Source\n  {}\n", e.source));
    out
}

/// `viv features <query>`: an area, one entry, or the entries a name finds;
/// failing that, a search.
pub fn lookup(q: &str) -> String {
    let c = catalog();
    let ql = q.trim().to_lowercase();
    if let Some((key, _, _)) = AREAS.iter().find(|(k, h, _)| *k == ql || h.to_lowercase() == ql) {
        return area_view(key);
    }
    let found = resolve_name(&c.entries, q);
    match found.len() {
        0 => {
            let s = search(q);
            format!("No entry is named `{q}`. Searching for it:\n\n{s}")
        }
        1 => detail(found[0]),
        _ => found.iter().map(|e| detail(e)).collect::<Vec<_>>().join("\n----\n\n"),
    }
}

/// The entries that match the words, best first, with a score: a word in the
/// id counts most, then the title, the keywords and forms, then the text.
pub fn search_entries(words: &str) -> Vec<(&'static Entry, usize)> {
    let c = catalog();
    let ws: Vec<String> = words.split_whitespace().map(|w| w.to_lowercase()).collect();
    if ws.is_empty() {
        return Vec::new();
    }
    let mut scored: Vec<(&Entry, usize)> = Vec::new();
    for e in &c.entries {
        let hay = e.haystack();
        let id = e.id.to_lowercase();
        let title = e.title.to_lowercase();
        let kw = e.keywords.join(" ").to_lowercase();
        let mut score = 0;
        let mut all = true;
        for w in &ws {
            let mut s = 0;
            if id == *w || id.split(|c: char| !c.is_alphanumeric() && c != '_').any(|p| p == w) {
                s += 8;
            } else if id.contains(w.as_str()) {
                s += 5;
            }
            if title.contains(w.as_str()) {
                s += 4;
            }
            if kw.split_whitespace().any(|k| k == w) {
                s += 4;
            }
            if hay.contains(w.as_str()) {
                s += 1;
            }
            if s == 0 {
                all = false;
            }
            score += s;
        }
        if all && score > 0 {
            // hand-written entries first among equals: they say when to use it
            scored.push((e, score * 2 + usize::from(e.hand())));
        }
    }
    scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.id.cmp(&b.0.id)));
    scored
}

/// `viv features --search <words>`.
pub fn search(words: &str) -> String {
    let found = search_entries(words);
    if found.is_empty() {
        return format!("nothing in the catalog mentions all of: {words}\n");
    }
    let mut out = String::new();
    let w = found.iter().take(25).map(|(e, _)| e.id.chars().count()).max().unwrap_or(0).min(34);
    for (e, _) in found.iter().take(25) {
        let line = if e.hand() { first_sentence(&e.what) } else { e.title.clone() };
        let line: String = if line.chars().count() > 90 { line.chars().take(87).collect::<String>() + "..." } else { line };
        out.push_str(&format!("  {:<w$}  {:<12} {}\n", e.id, e.area, line, w = w));
    }
    if found.len() > 25 {
        out.push_str(&format!("  ... and {} more\n", found.len() - 25));
    }
    out
}

// ---- JSON ----------------------------------------------------------------------

fn jarr(v: &[String]) -> String {
    format!("[{}]", v.iter().map(|s| json_str(s)).collect::<Vec<_>>().join(","))
}

/// One entry as JSON. With `run`, a generated example's output is computed.
pub fn entry_json(e: &Entry, run: bool) -> String {
    let forms = e.forms.iter().map(|f| format!("{{\"text\":{},\"note\":{}}}", json_str(&f.text), json_str(&f.note))).collect::<Vec<_>>().join(",");
    let examples = e
        .examples
        .iter()
        .map(|x| {
            let out = if run || x.output.is_some() { json_str(&example_output(x)) } else { "null".to_string() };
            format!("{{\"name\":{},\"code\":{},\"output\":{},\"cmd\":{}}}", json_str(&x.name), json_str(&example_code(x)), out, x.cmd.as_deref().map_or("null".to_string(), json_str))
        })
        .collect::<Vec<_>>()
        .join(",");
    let usage: Vec<String> = e.usage.iter().map(|d| d.text()).collect();
    format!(
        "{{\"id\":{},\"area\":{},\"group\":{},\"title\":{},\"what\":{},\"when\":{},\"forms\":[{}],\"examples\":[{}],\"no_example\":{},\"doc\":{},\"see\":{},\"keywords\":{},\"usage\":{},\"verb\":{},\"lens\":{},\"module\":{},\"interp_only\":{},\"memory\":{},\"source\":{},\"hand\":{}}}",
        json_str(&e.id),
        json_str(&e.area),
        json_str(&e.group),
        json_str(&e.title),
        json_str(&e.what),
        json_str(&e.when),
        forms,
        examples,
        json_str(&e.no_example),
        jarr(&e.doc),
        jarr(&e.see),
        jarr(&e.keywords),
        jarr(&usage),
        e.verb.as_deref().map_or("null".to_string(), json_str),
        e.lens.as_deref().map_or("null".to_string(), json_str),
        e.module.as_deref().map_or("null".to_string(), json_str),
        e.interp_only,
        json_str(&e.memory),
        json_str(&e.source),
        e.hand()
    )
}

/// `viv features --json`: the areas and every entry. Examples of builtins and
/// phrases carry their output only with `run`.
pub fn json(run: bool) -> String {
    let c = catalog();
    let areas = AREAS
        .iter()
        .map(|(k, h, w)| format!("{{\"key\":{},\"heading\":{},\"what\":{},\"count\":{}}}", json_str(k), json_str(h), json_str(w), c.entries.iter().filter(|e| e.area == *k).count()))
        .collect::<Vec<_>>()
        .join(",");
    let entries = c.entries.iter().map(|e| entry_json(e, run)).collect::<Vec<_>>().join(",\n");
    let hand = c.entries.iter().filter(|e| e.hand()).count();
    // features/builtins.txt as its groups of rows, for the reference page's tables
    let mut groups: Vec<(String, String, Vec<String>)> = Vec::new();
    for r in builtin_rows() {
        if groups.last().map_or(true, |g| g.0 != r.group) {
            groups.push((r.group.clone(), r.note.clone(), Vec::new()));
        }
        let row = jarr(&[r.name.clone(), r.sig.clone(), r.blurb.clone(), r.probe.clone()]);
        groups.last_mut().unwrap().2.push(row);
    }
    let builtins = groups.iter().map(|(t, n, rows)| format!("{{\"title\":{},\"note\":{},\"items\":[{}]}}", json_str(t), json_str(n), rows.join(","))).collect::<Vec<_>>().join(",\n");
    format!(
        "{{\"areas\":[{areas}],\"count\":{},\"hand\":{hand},\"generated\":{},\"errors\":{},\"builtin_groups\":[\n{builtins}\n],\"entries\":[\n{entries}\n]}}",
        c.entries.len(),
        c.entries.len() - hand,
        jarr(&c.errors)
    )
}

/// The wasm export's entry point: the arguments after `viv features`, as
/// one string, and the text the CLI would print.
pub fn cli_text(args: &str) -> String {
    let a: Vec<&str> = args.split_whitespace().collect();
    match a.as_slice() {
        [] => overview(),
        ["--json"] => json(false),
        ["--json", "--run"] | ["--run", "--json"] => json(true),
        ["--search", rest @ ..] if !rest.is_empty() => search(&rest.join(" ")),
        [q @ ..] if !q.iter().any(|w| w.starts_with("--")) => lookup(&q.join(" ")),
        _ => "usage: features [<name or area> | --search <words> | --json]\n".to_string(),
    }
}

// ---- modules.txt -----------------------------------------------------------------

/// A module's header: the title of its `###` block (`post:`) or the first
/// line of its leading comment, and the first paragraph after it.
pub fn module_header(text: &str) -> (String, String) {
    let lines: Vec<&str> = text.lines().collect();
    let mut title = String::new();
    let mut para: Vec<String> = Vec::new();
    if lines.first().map(|l| l.trim()) == Some("###") {
        let mut k = 1;
        // the fenced block's fields, up to its first blank line
        while k < lines.len() && !lines[k].trim().is_empty() && lines[k].trim() != "###" {
            if let Some(t) = lines[k].strip_prefix("post:") {
                title = t.trim().to_string();
            }
            k += 1;
        }
        while k < lines.len() && lines[k].trim().is_empty() {
            k += 1;
        }
        while k < lines.len() && !lines[k].trim().is_empty() && lines[k].trim() != "###" {
            para.push(lines[k].trim().to_string());
            k += 1;
        }
    } else {
        let mut k = 0;
        while k < lines.len() && (lines[k].starts_with("#!") || lines[k].trim().is_empty()) {
            k += 1;
        }
        while k < lines.len() && lines[k].starts_with('#') && !lines[k].starts_with("#!") {
            let l = lines[k].trim_start_matches('#').trim();
            if l.is_empty() {
                if !para.is_empty() {
                    break;
                }
            } else {
                para.push(l.to_string());
            }
            k += 1;
        }
        if !para.is_empty() {
            title = first_sentence(&para.join(" ")).trim_end_matches('.').to_string();
            if title.chars().count() > 90 {
                title = title.chars().take(87).collect::<String>() + "...";
            }
        }
    }
    let mut summary = para.join(" ");
    if summary.chars().count() > 400 {
        let cut: String = summary.chars().take(400).collect();
        summary = match cut.rfind(". ") {
            Some(i) if i > 100 => cut[..=i].to_string(),
            _ => cut + "...",
        };
    }
    (title, summary)
}

/// The top-level defs of a module, in order.
pub fn module_defs(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut fenced = false;
    for l in text.lines() {
        if l.trim() == "###" {
            fenced = !fenced;
            continue;
        }
        if fenced {
            continue;
        }
        let rest = l.strip_prefix("placeholder def ").or_else(|| l.strip_prefix("def ")).or_else(|| l.strip_prefix("contract "));
        if let Some(r) = rest {
            let name: String = r.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
            if !name.is_empty() && !out.contains(&name) {
                out.push(name);
            }
        }
    }
    out
}

/// features/modules.txt from the modules' paths (under the repository, no
/// `.viv`) and texts, sorted by path.
pub fn modules_text(mods: &[(String, String)]) -> String {
    let mut sorted: Vec<&(String, String)> = mods.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = String::from(
        "# The modules a program can import: lib/ and lang/viv/std. Written by\n# `viv features --write-modules` from each module's header; do not edit.\n# tests/features.rs fails when it is stale.\n",
    );
    for (path, text) in sorted {
        let (title, summary) = module_header(text);
        let defs = module_defs(text);
        let public: Vec<&String> = defs.iter().filter(|d| !d.starts_with('_')).collect();
        out.push_str(&format!("\n== {path}\n"));
        out.push_str(&format!("title: {}\n", if title.is_empty() { "(no header)" } else { &title }));
        if !summary.is_empty() {
            out.push_str(&format!("summary: {summary}\n"));
        }
        out.push_str(&format!("defs: {}\n", defs.len()));
        let shown: Vec<String> = public.iter().take(24).map(|s| s.to_string()).collect();
        let more = if public.len() > 24 { format!(", ... ({} more)", public.len() - 24) } else { String::new() };
        out.push_str(&format!("names: {}{more}\n", shown.join(", ")));
    }
    out
}

/// Which modules modules.txt lists: every `.viv` under lib/ and
/// lang/viv/std, except tests (a `tests` directory) and reference data the
/// tests read (`lib/hfref`).
pub fn is_catalog_module(rel: &str) -> bool {
    (rel.starts_with("lib/") || rel.starts_with("lang/viv/std/")) && rel.ends_with(".viv") && !rel.split('/').any(|p| p == "tests") && !rel.starts_with("lib/hfref/")
}

/// Every module modules.txt lists, as (path under the repository root
/// without `.viv`, text), read from the repository at `root`.
pub fn read_modules(root: &std::path::Path) -> Vec<(String, String)> {
    fn walk(p: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        if p.is_dir() {
            let mut es: Vec<std::path::PathBuf> = std::fs::read_dir(p).map(|r| r.flatten().map(|e| e.path()).collect()).unwrap_or_default();
            es.sort();
            for e in es {
                walk(&e, out);
            }
        } else if p.extension().is_some_and(|x| x == "viv") {
            out.push(p.to_path_buf());
        }
    }
    let mut files = Vec::new();
    for d in ["lib", "lang/viv/std"] {
        walk(&root.join(d), &mut files);
    }
    let mut out = Vec::new();
    for f in files {
        let Ok(rel) = f.strip_prefix(root) else { continue };
        let rel = rel.to_string_lossy().replace('\\', "/");
        if !is_catalog_module(&rel) {
            continue;
        }
        if let Ok(text) = std::fs::read_to_string(&f) {
            out.push((rel.trim_end_matches(".viv").to_string(), text));
        }
    }
    out
}

/// Areas' entry counts, for the report: (area, hand, generated).
pub fn counts() -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for e in &catalog().entries {
        *m.entry(e.area.clone()).or_insert(0) += 1;
    }
    m
}

/// The builtin names the interpreter's string, array and map modules define,
/// for tests that check the catalog against them (the modules stay private).
pub fn interp_module_builtins() -> Vec<String> {
    let mut out: Vec<String> = crate::interp::strings::BUILTINS.iter().map(|b| b.name.to_string()).collect();
    out.extend(crate::interp::arrays::SIGS.iter().map(|s| s.name.to_string()));
    out.extend(crate::interp::maps::NAMES.iter().map(|s| s.to_string()));
    out
}
