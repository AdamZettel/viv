// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Contracts and their methods: the declaration, the
//! signature checks, selection by call, by statement and by `--using`, every
//! method with its dissent check, and the engines agreeing on all of it.

use knotkit::span::SourceId;

fn run_compiled(src: &str, tag: &str) -> Option<String> {
    let c = viv::compile_c(src).expect("should compile");
    let dir = std::env::temp_dir().join(format!("viv_contract_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("p.c"), &c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let built = std::process::Command::new(&cc).args(["-std=c99", "-O2"]).arg(dir.join("p.c")).arg("-lm").arg("-o").arg(dir.join("p")).output().ok()?;
    assert!(built.status.success(), "generated C did not compile:\n{}", String::from_utf8_lossy(&built.stderr));
    let out = std::process::Command::new(dir.join("p")).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    Some(String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr))
}

fn interp(src: &str) -> String {
    match viv::interp::run(src) {
        Ok(out) => out.lines().filter(|l| !l.starts_with("__")).map(|l| format!("{l}\n")).collect(),
        Err(e) => format!("error: {e}\n"),
    }
}

fn agree(src: &str, tag: &str) -> String {
    let i = interp(src);
    if let Some(c) = run_compiled(src, tag) {
        assert_eq!(i.lines().next(), c.lines().next(), "the engines disagree:\n{i}\n---\n{c}");
        if !i.contains("error") {
            assert_eq!(i, c, "the engines disagree");
        }
    }
    i
}

fn parse_err(src: &str) -> String {
    match viv::parse(src, SourceId(0)) {
        Ok(_) => panic!("expected an error"),
        Err(ds) => ds.iter().map(|d| format!("{} {:?}", d.message, d.notes)).collect::<Vec<_>>().join("\n"),
    }
}

/// Two methods over a record and an array, returning two values; one takes
/// a setting.
const SUMSQ: &str = r#"record Scale {
    k: Num
}
def by_loop(x: Arr, s: Scale) {
    t = 0.0
    for i to len(x) {
        t = t + x[i] * x[i]
    }
    return s.k * t, len(x)
}
def by_dot(x: Arr, s: Scale, bias: Num) {
    return s.k * dot(x, x) + bias, len(x)
}
BIAS = 0.0
contract sumsq(x: Arr, s: Scale) -> total: Num, count: Int {
    intent "the sum of the squares of x, times s.k"
    methods by_loop, by_dot(bias = BIAS)
    agree total within relative 1.0e-12
    agree count exactly
}
"#;

#[test]
fn each_selection_runs_its_method_in_both_engines() {
    let src = format!(
        "{SUMSQ}sc = Scale {{ k: 2.0 }}\nt, n = sumsq([1.0, 2.0, 3.0], sc)\nshow t, n\nt2, n2 = sumsq([1.0, 2.0], sc) using by_dot\nshow t2, n2\nusing sumsq = every method\nt3, n3 = sumsq([1.0, 2.0, 3.0], sc)\nshow t3, n3\nshow using_sumsq == Sumsq.Every\n"
    );
    let out = agree(&src, "select");
    assert!(out.contains("t: 28  n: 3"), "{out}");
    assert!(out.contains("t2: 10  n2: 2"), "{out}");
    assert!(out.contains("t3: 28  n3: 3"), "{out}");
}

#[test]
fn every_method_stops_at_a_disagreement_and_names_it() {
    let src = SUMSQ.replace("BIAS = 0.0", "BIAS = 1.0") + "using sumsq = every method\nt, n = sumsq([1.0, 2.0, 3.0], Scale { k: 2.0 })\nshow t\n";
    let out = agree(&src, "dissent");
    assert!(out.contains("contract sumsq: by_dot disagrees with by_loop on total: 29 and 28 are 1 apart"), "{out}");
}

#[test]
fn a_normal_call_runs_the_default_only() {
    // by_dot gives another total, and nothing compares them
    let src = SUMSQ.replace("BIAS = 0.0", "BIAS = 1.0") + "t, n = sumsq([1.0, 2.0, 3.0], Scale { k: 2.0 })\nshow t\n";
    assert_eq!(agree(&src, "default"), "t: 28\n");
    // every method is every method in a release build too
    let every = SUMSQ.replace("BIAS = 0.0", "BIAS = 1.0") + "t, n = sumsq([1.0, 2.0, 3.0], Scale { k: 2.0 }) using every method\nshow t\n";
    assert!(viv::interp::run_release(&every).is_err());
}

#[test]
fn a_tolerance_is_one_number_or_name() {
    let formula = SUMSQ.replace("within relative 1.0e-12", "within relative 1.0e-12 * (1.0 + abs(total))");
    assert!(parse_err(&formula).contains("contract `sumsq`: a tolerance is one number or one name, not a formula"));
    let bare = SUMSQ.replace("within relative 1.0e-12", "within 1.0e-12");
    assert!(parse_err(&bare).contains("write `within relative <tolerance>` or `within absolute <tolerance>`"));
    // relative: |a - b| <= tol * max(1, |reference|); 28 vs 28 + 2e-11 passes at 1e-12
    let rel = SUMSQ.replace("BIAS = 0.0", "BIAS = 2.0e-11") + "t, n = sumsq([1.0, 2.0, 3.0], Scale { k: 2.0 }) using every method\nshow t\n";
    assert_eq!(agree(&rel, "rel"), "t: 28\n");
    let abs = rel.replace("within relative 1.0e-12", "within absolute 1.0e-12");
    assert!(agree(&abs, "abs").contains("by_dot disagrees with by_loop on total"));
}

#[test]
fn settings_have_defaults_and_a_settings_line_gives_them() {
    let base = "def m_a(x: Num, s: Num = 2.0) {\n    return x * s\n}\ndef m_b(x: Num) {\n    return x + x\n}\ncontract twice(x: Num) -> value: Num {\n    intent \"x doubled\"\n    methods m_a, m_b\n    agree value within absolute 1.0e-12\n}\n";
    let calls = "print(fmt(twice(3.0)))\nprint(fmt(twice(3.0) using m_b))\nsel = Twice.MB\nusing twice = sel\nprint(fmt(twice(3.0)))\n";
    assert_eq!(agree(&format!("{base}{calls}"), "defaults"), "6.0\n6.0\n6.0\n");
    // a settings line reaches every call that runs m_a, and none that runs m_b
    let k = 3.0;
    let set = format!("{base}factor = {k:?}\nsettings m_a(s = factor)\n{calls}");
    assert_eq!(agree(&set, "settings"), "9.0\n6.0\n6.0\n");
    assert!(parse_err(&format!("{base}settings m_a(t = 3.0)\n{calls}")).contains("`m_a` has no setting `t`; its settings are s"));
    assert!(parse_err(&format!("{base}settings m_b(s = 3.0)\n{calls}")).contains("`m_b` has no settings"));
    assert!(parse_err(&format!("{base}settings m_a(s = 3.0, s = 4.0)\n{calls}")).contains("`m_a`'s setting `s` is given twice"));
    assert!(parse_err(&format!("{base}print(twice(3.0) using m_a(s = 3.0))\n")).contains("`using` only chooses a method; give `m_a` its settings in a top-level `settings m_a(name = value)` statement"));
}

#[test]
fn every_method_of_a_group_runs_that_group_only() {
    let src = "def a1(x: Num) {\n    return x\n}\ndef b1(x: Num) {\n    return x + 1.0\n}\ndef b2(x: Num) {\n    return x + 1.0\n}\ncontract g(x: Num) -> value: Num {\n    intent \"x\"\n    methods\n        first: a1\n        second: b1, b2\n    agree value within absolute 1.0e-12\n}\nprint(fmt(g(1.0)))\nprint(fmt(g(1.0) using every second method))\nprint(fmt(g(1.0) using b2))\n";
    // b1 and b2 agree with each other; a1 differs, and is not run
    assert_eq!(agree(src, "group"), "1.0\n2.0\n2.0\n");
    let all = format!("{src}print(fmt(g(1.0) using every method))\n");
    assert!(interp(&all).contains("b1 disagrees with a1 on value"));
    assert!(parse_err(&src.replace("using every second method", "using every third method")).contains("contract `g` has no group `third`; its groups are first, second"));
    let set = viv::apply_sets(src, &[(format!("{}g", viv::USING_SET), "every-second".to_string())]).unwrap();
    assert!(set.contains("    using every second method\n}"), "{set}");
    let j = viv::contracts::json(&viv::contracts::scan(src)[0], 1);
    assert!(j.contains("\"groups\":[{\"label\":\"first\",\"methods\":[\"a1\"]},{\"label\":\"second\",\"methods\":[\"b1\",\"b2\"]}]"), "{j}");
}

#[test]
fn a_parameter_shadows_a_top_level_lambda_of_its_name() {
    let src = "def run(op, x: Num) {\n    return op(x)\n}\nop = fn(x) -> x + 100.0\nprint(fmt(run(fn(x) -> x * 2.0, 3.0)))\nprint(fmt(run(op, 3.0)))\n";
    assert_eq!(agree(src, "shadow"), "6.0\n103.0\n");
}

#[test]
fn the_median_is_the_consensus_when_asked() {
    let src = "def lo(x) {\n    return x - 1.0e-10\n}\ndef mid(x) {\n    return x\n}\ndef hi(x) {\n    return x + 2.0e-10\n}\ncontract near(x: Num) -> value: Num {\n    intent \"x\"\n    methods lo, mid, hi\n    agree value within absolute 1.0e-9\n    consensus median\n}\nprint(fmt(near(1.0) using every method))\nprint(fmt(near(1.0) using hi))\n";
    let out = agree(src, "median");
    assert_eq!(out, "1.0\n1.0000000002\n");
}

#[test]
fn a_method_that_does_not_fit_the_contract_is_refused() {
    let bad_type = SUMSQ.replace("def by_dot(x: Arr, s: Scale, bias: Num)", "def by_dot(x: Arr, s: Num, bias: Num)");
    let e = parse_err(&bad_type);
    assert!(e.contains("`by_dot` takes `s: Num` as parameter 2, where contract `sumsq` passes `s: Scale`"), "{e}");
    let unbound = SUMSQ.replace("by_dot(bias = BIAS)", "by_dot");
    let e = parse_err(&unbound);
    assert!(e.contains("`by_dot` takes bias beyond the contract's parameters with no default"), "{e}");
    let wrong_bind = SUMSQ.replace("by_dot(bias = BIAS)", "by_dot(offset = BIAS)");
    assert!(parse_err(&wrong_bind).contains("`by_dot` has no parameter `offset` beyond the contract's (bias)"));
    let missing = SUMSQ.replace("methods by_loop, by_dot(bias = BIAS)", "methods by_loop, by_dott");
    assert!(parse_err(&missing).contains("contract `sumsq` lists `by_dott`, which is not a def of the program"));
    let returns = SUMSQ.replace("return s.k * t, len(x)", "return s.k * t");
    assert!(parse_err(&returns).contains("`by_loop` returns 1 value; contract `sumsq` has 2 outputs (total, count)"));
    let no_agree = SUMSQ.replace("    agree total within relative 1.0e-12\n    agree count exactly\n", "");
    assert!(parse_err(&no_agree).contains("contract `sumsq` has 2 methods and no `agree` line"));
    let int_tol = SUMSQ.replace("agree count exactly", "agree count within absolute 1.0");
    assert!(parse_err(&int_tol).contains("`count` is Int; write `agree count exactly`"));
    let no_intent = SUMSQ.replace("    intent \"the sum of the squares of x, times s.k\"\n", "");
    assert!(parse_err(&no_intent).contains("contract `sumsq` has no `intent` line"));
    let bad_using = format!("{SUMSQ}t, n = sumsq([1.0], Scale {{ k: 1.0 }}) using by_cross\n");
    assert!(parse_err(&bad_using).contains("contract `sumsq` has no method `by_cross`; its methods are by_loop, by_dot (or `every method`, or a variable holding the method)"));
    let inner = format!("{SUMSQ}def f() {{\n    using sumsq = by_dot\n    return 1.0\n}}\n");
    assert!(parse_err(&inner).contains("`using sumsq = ...` is a top-level statement"));
}

#[test]
fn using_on_the_command_line_sets_the_start() {
    let src = format!("{SUMSQ}t, n = sumsq([1.0, 2.0], Scale {{ k: 1.0 }})\nshow t\n");
    let set = viv::apply_sets(&src, &[(format!("{}sumsq", viv::USING_SET), "by_dot".to_string())]).unwrap();
    assert!(set.contains("    using by_dot\n}"), "{set}");
    let every = viv::apply_sets(&set, &[(format!("{}sumsq", viv::USING_SET), "every".to_string())]).unwrap();
    assert!(every.contains("    using every method\n}") && !every.contains("using by_dot"), "{every}");
    assert_eq!(interp(&every), "t: 5\n");
    let e = viv::apply_sets(&src, &[(format!("{}sumsq", viv::USING_SET), "by_cross".to_string())]).unwrap_err();
    assert_eq!(e, "--using sumsq=by_cross: contract `sumsq` has the methods by_loop, by_dot (or `every`)");
    let e = viv::apply_sets(&src, &[(format!("{}cube", viv::USING_SET), "x".to_string())]).unwrap_err();
    assert_eq!(e, "--using cube=x: the program has no contract `cube`; its contracts are sumsq");
}

#[test]
fn the_tools_read_the_declaration() {
    let cs = viv::contracts::scan(SUMSQ);
    assert_eq!(cs.len(), 1);
    let j = viv::contracts::json(&cs[0], 18);
    assert!(j.contains("\"name\":\"sumsq\""), "{j}");
    assert!(j.contains("\"methods\":[\"by_loop\",\"by_dot\"]"), "{j}");
    assert!(j.contains("\"default\":\"by_loop\",\"using\":\"by_loop\""), "{j}");
    assert!(j.contains("{\"output\":\"count\",\"within\":null}"), "{j}");
    assert!(j.contains("\"selector\":\"using_sumsq\""), "{j}");
}

#[test]
fn an_uncalled_registry_def_does_not_stop_the_program() {
    // `sweep` in std/contracts.viv takes a registry and nothing here calls it
    let src = "def sq(x) {\n    return x * x\n}\ndef run_all(ms, x) {\n    s = 0.0\n    for i to len(ms) {\n        s = s + apply(ms[i], x)\n    }\n    return s\n}\ndef outer(ms, x) {\n    return run_all(ms, x)\n}\nprint(sq(3.0))\n";
    assert_eq!(agree(src, "uncalled"), "9\n");
}
