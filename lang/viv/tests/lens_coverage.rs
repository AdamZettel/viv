// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Intent coverage, the ratchet, needed-ness and `serves`.
//!
//! - coverage counts defs with a `what`, steps with text, variables with a
//!   glossary line, and stale entries, from either home;
//! - the ratchet file round-trips, and a drop in any share or a rise in the
//!   stale, dead or broken-serves count is a failure while a gain is not;
//! - a def is needed when a program's top level, a test, an entry point
//!   (`entry` in the card or `#~ entry`) or a registry reaches it; recursion
//!   alone does not make a def needed;
//! - `serves` citations are checked to exist, and `lens check` holds the
//!   syntax of `entry` and `serves`;
//! - a long def explained with `=` lines passes the steps rule.

use viv::lens::{self, Input, IntentFile, ModuleText};
use viv::lens_cov::{self, Count, Known, Recorded, RepoFile};

fn hash_of(src: &str, name: &str) -> String {
    viv::def_hashes(src).unwrap().into_iter().find(|h| h.name == name).unwrap().body_short
}

fn load(src: &str, intent: Option<&str>) -> lens::Loaded {
    let input = Input {
        root: ModuleText { name: "fx/m".into(), path: "fx/m.viv".into(), src: src.to_string(), intent_path: "fx/m.intent".into(), intent: intent.map(String::from) },
        used: vec![],
        expanded: src.to_string(),
        companion_tests: vec![],
    };
    lens::load(&input).unwrap()
}

const MODULE: &str = r#"
#| Add one.
def inc(x) {
    y = x + 1
    return y
}

def twice(x) {
    step first "apply inc once" {
        a = inc(x)
    }
    step second {
        b = inc(a)
    }
    return b
}

test "inc adds one" {
    assert_near(inc(1), 2, 0)
}
"#;

#[test]
fn coverage_counts_defs_steps_glossary_and_stale_entries() {
    let c = lens_cov::coverage(&load(MODULE, None));
    // inc has a what inline; twice has none
    assert_eq!(c.defs, Count { done: 1, total: 2 });
    // `first` has text in the code, `second` has none anywhere
    assert_eq!(c.steps, Count { done: 1, total: 2 });
    // x, y of inc and x, a, b of twice; nothing documented
    assert_eq!(c.glossary, Count { done: 0, total: 5 });
    // inc's inline what and twice's inline step text are unstamped
    assert_eq!(c.stale, 2);
    let kinds: Vec<String> = c.items.iter().map(|i| format!("{} {}", i.kind, i.anchor)).collect();
    assert_eq!(kinds, ["what twice", "step-text twice/second", "unstamped inc", "unstamped twice"]);

    // the file supplies the rest; its stamps are current
    let intent = format!(
        "module fx/m\n  A fixture.\n\ninc  @{}\n  why   a def to call\n  x     the number\n  y     one more than x\n\ntwice  @{}\n  what  Add two.\n  x     the number\n  a     x plus one\n  b     x plus two\n\n  first\n  second  apply inc again\n  return b\n",
        hash_of(MODULE, "inc"),
        hash_of(MODULE, "twice")
    );
    let l = load(MODULE, Some(&intent));
    let c = lens_cov::coverage(&l);
    assert_eq!((c.defs, c.steps, c.glossary, c.stale), (Count { done: 2, total: 2 }, Count { done: 2, total: 2 }, Count { done: 5, total: 5 }, 0), "{:?}", c.items);
    assert!(lens::check(&l, &mut |_, _| None).is_empty());

    // an edited body makes the entry stale, and the share of defs stays
    let edited = MODULE.replace("y = x + 1", "y = 1 + x");
    let c = lens_cov::coverage(&load(&edited, Some(&intent)));
    assert_eq!(c.stale, 1);
    assert_eq!(c.defs, Count { done: 2, total: 2 });
}

#[test]
fn the_ratchet_round_trips_and_fails_only_on_a_drop() {
    let r = Recorded { defs: Count { done: 3, total: 4 }, steps: Count { done: 0, total: 0 }, glossary: Count { done: 10, total: 40 }, stale: 2, dead: 1, serves: 0 };
    let mods = vec![("lib/b".to_string(), r), ("lib/a".to_string(), Recorded::default())];
    let text = lens_cov::ratchet_text(&mods);
    let back = lens_cov::parse_ratchet(&text).unwrap();
    // sorted by name, every number kept
    assert_eq!(back, vec![("lib/a".to_string(), Recorded::default()), ("lib/b".to_string(), r)]);
    assert_eq!(lens_cov::ratchet_text(&back), text, "printing what was parsed gives the same text");

    let (drops, gains) = lens_cov::ratchet_compare("lib/b", &r, &r);
    assert!(drops.is_empty() && gains.is_empty());
    // one more def with no what: 3/5 < 3/4
    let worse = Recorded { defs: Count { done: 3, total: 5 }, ..r };
    assert_eq!(lens_cov::ratchet_compare("lib/b", &worse, &r).0.len(), 1);
    // the same share over more defs is not a drop: 6/8 == 3/4
    let same = Recorded { defs: Count { done: 6, total: 8 }, ..r };
    assert!(lens_cov::ratchet_compare("lib/b", &same, &r).0.is_empty());
    // a new stale entry, a new dead def, a broken serves: each a drop
    for bad in [Recorded { stale: 3, ..r }, Recorded { dead: 2, ..r }, Recorded { serves: 1, ..r }] {
        assert_eq!(lens_cov::ratchet_compare("lib/b", &bad, &r).0.len(), 1, "{bad:?}");
    }
    // a gain is reported for `--update`, and is not a failure
    let better = Recorded { glossary: Count { done: 20, total: 40 }, stale: 0, ..r };
    let (drops, gains) = lens_cov::ratchet_compare("lib/b", &better, &r);
    assert!(drops.is_empty());
    assert_eq!(gains.len(), 2);
    // a module with nothing to cover that gains an undocumented def has dropped
    let empty = Recorded { steps: Count { done: 0, total: 0 }, ..r };
    let grown = Recorded { steps: Count { done: 0, total: 1 }, ..r };
    assert_eq!(lens_cov::ratchet_compare("lib/b", &grown, &empty).0.len(), 1);
}

fn file(name: &str, src: &str, intent: Option<&str>) -> RepoFile {
    // the fixture's modules are all under lib/
    let uses = viv::imports::known_refs(name, src, &|c: &str| c.strip_prefix("lib/").is_some_and(|r| !r.contains('/'))).into_iter().map(|(m, _)| m).collect();
    RepoFile { name: name.into(), path: format!("{name}.viv"), src: src.into(), intent_path: format!("{name}.intent"), intent: intent.map(String::from), uses }
}

const LIB: &str = r#"
def used_by_program(x) {
    return helper(x)
}
def helper(x) {
    return x + 1
}
def used_by_test(x) {
    return x * 2
}
def member_a(x) {
    return x
}
def member_b(x) {
    return -x
}
def registry() {
    return [member_a, member_b]
}
def declared(x) {
    return x
}
#~ entry
def tagged(x) {
    return x
}
def lonely(x) {
    return x
}
def self_calls(n) {
    if n > 0 {
        return self_calls(n - 1)
    }
    return 0
}
def ping(n) {
    return pong(n)
}
def pong(n) {
    return ping(n)
}
def countdown(n) {
    if n > 0 {
        return countdown(n - 1)
    }
    return 0
}
def serving(x) {
    return x
}
test "used_by_test doubles" {
    assert_near(used_by_test(2), 4, 0)
}
"#;

const LIB_INTENT: &str = "module lib/m\n  A fixture.\n\ndeclared\n  entry  the site's workbench, which calls it by name\n\nserving\n  serves  [test \"used_by_test doubles\"] [test \"no such test\"] [post \"a-post\"] [post \"gone\"]\n";

const PROG: &str = r#"from m import countdown, registry, used_by_program
r = registry()
print(apply(r[0], 1.0))
print(used_by_program(1.0))
print(countdown(3))
"#;

#[test]
fn dead_code_is_what_no_program_test_entry_or_registry_reaches() {
    let files = vec![file("lib/m", LIB, Some(LIB_INTENT)), file("prog/p", PROG, None)];
    let mut known = Known::default();
    known.tests.insert("used_by_test doubles".into());
    known.posts.insert("a-post".into());
    let n = lens_cov::needed(&files, &known, &["prog/p".to_string()]);
    let dead: Vec<&str> = n.dead_in("lib/m").iter().map(|d| d.name.as_str()).collect();
    // the program reaches used_by_program and, through it, helper; the test
    // reaches used_by_test; the registry reaches both members; declared and
    // tagged are entry points; countdown is recursive and reached
    assert_eq!(dead, ["lonely", "self_calls", "ping", "pong", "serving"]);
    // callers are computed from the code
    let h = n.get("lib/m", "helper").unwrap();
    assert_eq!(h.callers, ["lib/m.used_by_program"]);
    assert!(h.on_story, "the story program reaches helper");
    assert!(!n.get("lib/m", "used_by_test").unwrap().on_story);
    let a = n.get("lib/m", "member_a").unwrap();
    assert_eq!(a.callers, ["lib/m.registry"], "a registry names its members");
    assert!(n.get("lib/m", "declared").unwrap().entry && n.get("lib/m", "tagged").unwrap().entry);
    // the top level of a file is a caller too
    assert_eq!(n.get("lib/m", "used_by_program").unwrap().callers, ["prog/p.<main>"]);
    // with no program at all, only the test and the entry points hold
    let alone = lens_cov::needed(&files[..1], &known, &[]);
    let live: Vec<&str> = {
        let mut v: Vec<&str> = alone.defs.values().filter(|d| d.reachable).map(|d| d.name.as_str()).collect();
        v.sort();
        v
    };
    assert_eq!(live, ["declared", "tagged", "used_by_test"]);
}

#[test]
fn serves_names_something_that_exists() {
    let files = vec![file("lib/m", LIB, Some(LIB_INTENT)), file("prog/p", PROG, None)];
    let mut known = Known::default();
    known.tests.insert("used_by_test doubles".into());
    known.posts.insert("a-post".into());
    let n = lens_cov::needed(&files, &known, &[]);
    let bad: Vec<String> = n.bad_serves_in("lib/m").iter().map(|(d, k, t)| format!("{} {k} {t}", d.name)).collect();
    assert_eq!(bad, ["serving test no such test", "serving post gone"]);
    // the module's coverage carries the dead defs and the broken citations
    let mut c = lens_cov::ModCov { name: "lib/m".into(), ..Default::default() };
    lens_cov::add_needed(&mut c, &n);
    assert_eq!((c.dead, c.serves), (5, 2));
    // a serves line is citations, and nothing else
    assert_eq!(lens_cov::serves_targets("[test \"a\"] [program \"scf/story_water\"]"), (vec![("test".into(), "a".into()), ("program".into(), "scf/story_water".into())], String::new()));
    assert_eq!(lens_cov::serves_targets("the water page [nonsense \"x\"]").1, "the water page [nonsense \"x\"]");
}

#[test]
fn entry_and_serves_are_card_fields() {
    let text = "module fx/m\n\ninc  @00000000\n  what    Add one.\n  entry   the workbench's examples\n  serves  [test \"inc adds one\"]\n  x       the number\n";
    let f = IntentFile::parse(text).unwrap();
    assert_eq!(f.print(), text, "the file round-trips byte for byte");
    let (_, es) = f.model();
    assert_eq!(es[0].field("entry").unwrap().text, "the workbench's examples");
    assert_eq!(es[0].field("serves").unwrap().text, "[test \"inc adds one\"]");
    assert_eq!(es[0].gloss.iter().map(|g| g.key.as_str()).collect::<Vec<_>>(), ["x"], "entry and serves are not glossary names");

    // lens check holds their syntax: an entry says who calls it, and serves is citations
    let h = hash_of(MODULE, "inc");
    let bad = format!("inc  @{h}\n  entry\n  serves  the tests\n\ntwice  @{}\n  what  Add two.\n\n  first\n  second  apply inc again\n  return b\n", hash_of(MODULE, "twice"));
    let fs = lens::check(&load(MODULE, Some(&bad)), &mut |_, _| None);
    let kinds: Vec<String> = fs.iter().map(|f| format!("{} {}", f.kind, f.anchor)).collect();
    assert_eq!(kinds, ["entry inc", "serves inc"]);
    // and the prose rules do not read them: a number in a test's name is not an uncited number
    let good = bad.replace("  entry\n  serves  the tests\n", "  entry   the workbench\n  serves  [test \"inc adds 1 to 1\"]\n");
    let fs = lens::check(&load(MODULE, Some(&good)), &mut |_, _| None);
    assert!(fs.is_empty(), "{}", fs.iter().map(|f| f.render()).collect::<String>());
}

#[test]
fn a_long_def_explained_line_by_line_passes_the_steps_rule() {
    let body: String = (0..30).map(|k| format!("    s = s + {k}\n")).collect();
    let src = format!("def long(n) {{\n    s = 0\n{body}    t = s * n\n    return t\n}}\n");
    let h = hash_of(&src, "long");
    let bare = format!("long  @{h}\n  what  Sum and scale.\n");
    let kinds: Vec<String> = lens::check(&load(&src, Some(&bare)), &mut |_, _| None).iter().map(|f| f.kind.to_string()).collect();
    assert_eq!(kinds, ["steps"]);
    let anchored = format!("{bare}\n  = t  the sum, scaled by n\n");
    let fs = lens::check(&load(&src, Some(&anchored)), &mut |_, _| None);
    assert!(fs.is_empty(), "{}", fs.iter().map(|f| f.render()).collect::<String>());
}

#[test]
fn a_program_reports_what_its_modules_define_and_it_does_not_reach() {
    let files = vec![file("lib/m", LIB, Some(LIB_INTENT)), file("prog/p", PROG, None)];
    let (total, un) = lens_cov::unreached_by(&files, "prog/p");
    assert_eq!(total, 14);
    let names: Vec<&str> = un.iter().map(|(_, d)| d.as_str()).collect();
    // the library's own test and entry points do not count for this program
    assert!(names.contains(&"used_by_test") && names.contains(&"declared") && names.contains(&"tagged"), "{names:?}");
    assert!(!names.contains(&"helper") && !names.contains(&"member_b"));
}

#[test]
fn debt_puts_what_a_run_calls_first() {
    let out = "3\n__graph__ {\"nodes\":[\"<main>\",\"used_by_program\",\"helper\"],\"edges\":[{\"from\":\"<main>\",\"to\":\"used_by_program\",\"count\":1},{\"from\":\"used_by_program\",\"to\":\"helper\",\"count\":7},{\"from\":\"<main>\",\"to\":\"helper\",\"count\":2}]}\n";
    let calls = lens_cov::run_calls(out);
    assert_eq!(calls.get("helper"), Some(&9), "calls into a def are summed over its callers");
    assert_eq!(calls.get("used_by_program"), Some(&1));

    let files = vec![file("lib/m", LIB, None), file("prog/p", PROG, None)];
    let n = lens_cov::needed(&files, &Known::default(), &[]);
    let mut c = lens_cov::ModCov { name: "lib/m".into(), ..Default::default() };
    for (k, d) in [("what", "lonely"), ("what", "used_by_program"), ("what", "helper")] {
        c.items.push(lens_cov::Owed { kind: k, def: d.into(), anchor: d.into(), path: "lib/m.viv".into(), line: 1, message: String::new() });
    }
    let order = |t: String| -> Vec<String> { t.lines().filter(|l| l.starts_with("  what")).map(|l| l.split_whitespace().nth(1).unwrap().to_string()).collect() };
    // statically: by callers (helper and used_by_program have one each, lonely none)
    assert_eq!(order(lens_cov::debt_text(&[c.clone()], &n, None, None)).last().unwrap(), "lonely");
    // with a run: the most called first
    assert_eq!(order(lens_cov::debt_text(&[c], &n, None, Some(&calls))), ["helper", "used_by_program", "lonely"]);
}
