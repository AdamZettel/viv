// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// REUSE-IgnoreStart (this file reads and writes the tags; it does not use them)
//! The licenses of an exported program (`viv export`, and `vipyr.compile`
//! through it). Every source module that contributes code says its license
//! in an SPDX header; the runtime C the backend adds has its own. The
//! exported C file starts with a comment that lists each component, its
//! SPDX license and its copyright, and a `NOTICES` file beside it holds the
//! full texts. A GPL or AGPL module refuses the build; an MPL or LGPL one
//! is allowed, with a warning that names the files whose source must stay
//! available. LICENSING.md, "Exported C".

use std::fmt::Write as _;

/// One part of an exported program and its license.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Component {
    /// the module's path, or the name of a runtime part
    pub name: String,
    /// an SPDX license expression; `NOASSERTION` when the module says none
    pub license: String,
    pub copyright: Vec<String>,
}

/// The license texts the compiler carries, so that an export outside a
/// checkout still writes them.
const TEXTS: &[(&str, &str)] = &[
    ("MIT", include_str!("../../../LICENSES/MIT.txt")),
    ("Apache-2.0", include_str!("../../../LICENSES/Apache-2.0.txt")),
    ("MPL-2.0", include_str!("../../../LICENSES/MPL-2.0.txt")),
    ("SunPro", include_str!("../../../LICENSES/SunPro.txt")),
    ("blessing", include_str!("../../../LICENSES/blessing.txt")),
    ("BSD-3-Clause", include_str!("../../../LICENSES/BSD-3-Clause.txt")),
    // lib/minpack; its license asks for the notice to be reproduced
    ("LicenseRef-Minpack", include_str!("../../../LICENSES/LicenseRef-Minpack.txt")),
];

/// The SPDX tags of a file's text: its license expression (several tags
/// joined with AND) and its copyright lines.
pub fn spdx_of(text: &str) -> (Option<String>, Vec<String>) {
    let mut lic: Vec<String> = Vec::new();
    let mut cop = Vec::new();
    for line in text.lines() {
        let tag = |key: &str| line.find(key).map(|i| clean(&line[i + key.len()..]));
        if let Some(v) = tag("SPDX-License-Identifier:") {
            if !v.is_empty() && !lic.contains(&v) {
                lic.push(v);
            }
        } else if let Some(v) = tag("SPDX-FileCopyrightText:") {
            if !v.is_empty() {
                cop.push(v);
            }
        }
    }
    let expr = match lic.len() {
        0 => None,
        1 => Some(lic.remove(0)),
        _ => Some(lic.iter().map(|l| paren(l)).collect::<Vec<_>>().join(" AND ")),
    };
    (expr, cop)
}

fn clean(s: &str) -> String {
    s.trim().trim_end_matches("*/").trim_end_matches("-->").trim().to_string()
}

fn paren(l: &str) -> String {
    if l.contains(' ') { format!("({l})") } else { l.to_string() }
}

/// The license ids of an expression.
pub fn ids(expr: &str) -> Vec<String> {
    expr.split(|c: char| c.is_whitespace() || c == '(' || c == ')')
        .filter(|t| !t.is_empty() && !matches!(*t, "AND" | "OR" | "WITH"))
        .map(str::to_string)
        .collect()
}

/// The modules of a program, (name, text), each once, as components. A
/// module without a header gets what `fallback(name)` says (REUSE.toml).
pub fn gather(sources: &[(String, String)], fallback: &dyn Fn(&str) -> Option<(String, Vec<String>)>) -> Vec<Component> {
    let mut out: Vec<Component> = Vec::new();
    for (name, text) in sources {
        if out.iter().any(|c| &c.name == name) {
            continue;
        }
        let (mut lic, mut copyright) = spdx_of(text);
        if lic.is_none() {
            if let Some((l, c)) = fallback(name) {
                lic = Some(l);
                if copyright.is_empty() {
                    copyright = c;
                }
            }
        }
        out.push(Component { name: name.clone(), license: lic.unwrap_or_else(|| "NOASSERTION".into()), copyright });
    }
    out
}

/// The license of `file` from the REUSE.toml of the nearest directory
/// above it that has one: the last annotation whose path matches (REUSE
/// spec 3.x; LICENSING.md). Reads the one-line strings and arrays that
/// Viv's REUSE.toml uses.
pub fn reuse_toml(file: &std::path::Path) -> Option<(String, Vec<String>)> {
    let file = std::fs::canonicalize(file).ok()?;
    let mut dir = file.parent()?;
    let text = loop {
        if let Ok(t) = std::fs::read_to_string(dir.join("REUSE.toml")) {
            break t;
        }
        dir = dir.parent()?;
    };
    let rel = file.strip_prefix(dir).ok()?.to_string_lossy().replace('\\', "/");
    let mut hit = None;
    let mut cur: Option<(Vec<String>, String, Vec<String>)> = None;
    let finish = |cur: &mut Option<(Vec<String>, String, Vec<String>)>, hit: &mut Option<(String, Vec<String>)>| {
        if let Some((paths, lic, cop)) = cur.take() {
            if !lic.is_empty() && paths.iter().any(|p| glob_match(p, &rel)) {
                *hit = Some((lic, cop));
            }
        }
    };
    for line in text.lines().map(str::trim) {
        if line == "[[annotations]]" {
            finish(&mut cur, &mut hit);
            cur = Some((Vec::new(), String::new(), Vec::new()));
            continue;
        }
        let (Some(c), Some((k, v))) = (cur.as_mut(), line.split_once('=')) else { continue };
        let vals: Vec<String> = v.split('"').skip(1).step_by(2).map(str::to_string).collect();
        match k.trim() {
            "path" => c.0 = vals,
            "SPDX-License-Identifier" => c.1 = vals.join(" AND "),
            "SPDX-FileCopyrightText" => c.2 = vals,
            _ => {}
        }
    }
    finish(&mut cur, &mut hit);
    hit
}

/// REUSE.toml's globs: `*` within a directory, `**` across them.
pub fn glob_match(pat: &str, path: &str) -> bool {
    fn go(p: &[u8], s: &[u8]) -> bool {
        match p.first() {
            None => s.is_empty(),
            Some(b'*') if p.get(1) == Some(&b'*') => (0..=s.len()).any(|i| go(&p[2..], &s[i..])),
            Some(b'*') => (0..=s.len()).take_while(|&i| i == 0 || s[i - 1] != b'/').any(|i| go(&p[1..], &s[i..])),
            Some(&c) => s.first() == Some(&c) && go(&p[1..], &s[1..]),
        }
    }
    go(pat.as_bytes(), path.as_bytes())
}

/// What the C backend adds to every exported program, and SQLite when the
/// program uses the db_* builtins.
pub fn runtime(sqlite: bool) -> Vec<Component> {
    let mut out = vec![
        Component { name: "the Viv runtime (lang/viv/src)".into(), license: "MIT OR Apache-2.0".into(), copyright: vec!["2026 The Viv authors".into()] },
        Component {
            name: "Viv's libm (lang/viv/src/vmath.c: fdlibm as musl carries it)".into(),
            license: "MIT AND SunPro".into(),
            copyright: vec!["1993 Sun Microsystems, Inc.".into(), "2005-2020 Rich Felker et al. (musl)".into(), "2026 The Viv authors".into()],
        },
    ];
    if sqlite {
        out.push(Component { name: "SQLite (sqlite3.c, beside this file)".into(), license: "blessing".into(), copyright: vec!["NONE (the SQLite authors dedicate the code to the public domain)".into()] });
    }
    out
}

/// What a license asks of whoever ships the exported program.
#[derive(Debug, PartialEq, Eq)]
pub enum Kind {
    Permissive,
    /// MPL, LGPL: the files keep their license, their source stays available
    FileCopyleft,
    /// GPL, AGPL: the whole program would fall under it
    Copyleft,
}

pub fn kind(id: &str) -> Kind {
    if id.starts_with("GPL-") || id.starts_with("AGPL-") {
        Kind::Copyleft
    } else if id.starts_with("MPL-") || id.starts_with("LGPL-") {
        Kind::FileCopyleft
    } else {
        Kind::Permissive
    }
}

/// The components' kind: an expression is as strict as its least strict
/// choice under OR, its strictest part under AND. Treated simply: an
/// expression with OR and a permissive choice is permissive.
fn strictest(expr: &str) -> Kind {
    let ids = ids(expr);
    if expr.contains(" OR ") && ids.iter().any(|i| kind(i) == Kind::Permissive) {
        return Kind::Permissive;
    }
    if ids.iter().any(|i| kind(i) == Kind::Copyleft) {
        Kind::Copyleft
    } else if ids.iter().any(|i| kind(i) == Kind::FileCopyleft) {
        Kind::FileCopyleft
    } else {
        Kind::Permissive
    }
}

/// Refuse a GPL module; warn about MPL and LGPL ones. A module with no
/// header (a user's own program, usually) is listed as NOASSERTION.
pub fn check(cs: &[Component]) -> Result<Vec<String>, String> {
    let gpl: Vec<&Component> = cs.iter().filter(|c| strictest(&c.license) == Kind::Copyleft).collect();
    if !gpl.is_empty() {
        let names: Vec<String> = gpl.iter().map(|c| format!("{} ({})", c.name, c.license)).collect();
        return Err(format!(
            "error: the program imports GPL-licensed code: {}. Viv's exports are permissive libraries, and GPL code \
             would put the whole program under the GPL; `viv export` refuses it. Ask the module's authors for a \
             permissive or LGPL license, or leave the module out (LICENSING.md, \"Exported C\").",
            names.join(", ")
        ));
    }
    let mut warnings = Vec::new();
    for c in cs.iter().filter(|c| strictest(&c.license) == Kind::FileCopyleft) {
        warnings.push(format!(
            "warning: {} is {}: whoever ships this program must keep that file's source available under {} \
             (its changes too); the rest of the program keeps its own license. NOTICES has the text.",
            c.name, c.license, c.license
        ));
    }
    Ok(warnings)
}

/// The SPDX expression of the whole exported file.
pub fn expression(cs: &[Component]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for c in cs {
        if c.license != "NOASSERTION" && !parts.contains(&paren(&c.license)) {
            parts.push(paren(&c.license));
        }
    }
    if parts.len() == 1 { parts[0].trim_start_matches('(').trim_end_matches(')').to_string() } else { parts.join(" AND ") }
}

/// The comment at the top of the exported C file.
pub fn c_comment(cs: &[Component]) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "/* SPDX-License-Identifier: {}", expression(cs));
    let mut copies: Vec<&String> = Vec::new();
    for c in cs {
        for line in &c.copyright {
            if !copies.contains(&line) {
                copies.push(line);
            }
        }
    }
    for line in copies {
        let _ = writeln!(s, "   SPDX-FileCopyrightText: {}", line.replace("*/", "* /"));
    }
    let _ = writeln!(s, "\n   This file is compiled from these components; NOTICES, beside it, has\n   the full text of each license.\n");
    for c in cs {
        let _ = writeln!(s, "     {}\n         {}{}", c.name.replace("*/", "* /"), c.license, if c.copyright.is_empty() { String::new() } else { format!("; {}", c.copyright.join("; ").replace("*/", "* /")) });
    }
    s.push_str("*/\n");
    s
}

/// The NOTICES file: the components, then each license's text once.
/// `extra(id)` finds a text the compiler does not carry (LICENSES/ of the
/// checkout).
pub fn notices(cs: &[Component], extra: &dyn Fn(&str) -> Option<String>) -> String {
    let mut s = String::from("Notices for this exported program (written by `viv export`)\n\n");
    for c in cs {
        let _ = writeln!(s, "{}\n    License: {}", c.name, c.license);
        for line in &c.copyright {
            let _ = writeln!(s, "    Copyright: {line}");
        }
        s.push('\n');
    }
    let mut seen: Vec<String> = Vec::new();
    for c in cs {
        for id in ids(&c.license) {
            if seen.contains(&id) || id == "NOASSERTION" {
                continue;
            }
            seen.push(id.clone());
            let text = TEXTS.iter().find(|(k, _)| *k == id).map(|(_, t)| t.to_string()).or_else(|| extra(&id));
            let _ = writeln!(s, "{}\n{id}\n\n{}\n", "=".repeat(72), text.unwrap_or_else(|| missing(&id)));
        }
    }
    s
}

/// What NOTICES says for a license whose text neither the compiler nor
/// the program's LICENSES/ holds. A LicenseRef id is not on spdx.org.
fn missing(id: &str) -> String {
    if id.starts_with("LicenseRef-") {
        format!("(the text of {id} is not in this compiler; copy it from the LICENSES/ folder of the source it came from)")
    } else {
        format!("(the text of {id} is not in this compiler; see https://spdx.org/licenses/{id}.html)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_are_read_and_joined() {
        let (l, c) = spdx_of("x = 1\n# SPDX-License-Identifier: MIT\n/* SPDX-License-Identifier: SunPro */\n# SPDX-FileCopyrightText: 2026 A\n");
        assert_eq!(l.as_deref(), Some("MIT AND SunPro"));
        assert_eq!(c, vec!["2026 A".to_string()]);
        assert_eq!(spdx_of("nothing").0, None);
    }

    #[test]
    fn kinds_follow_the_expression() {
        let c = |l: &str| Component { name: "m.viv".into(), license: l.into(), copyright: vec![] };
        assert!(check(&[c("MIT OR Apache-2.0")]).unwrap().is_empty());
        assert!(check(&[c("GPL-2.0-only")]).is_err());
        assert!(check(&[c("MIT AND AGPL-3.0-only")]).is_err());
        assert!(check(&[c("MIT OR GPL-3.0-only")]).unwrap().is_empty());
        assert_eq!(check(&[c("LGPL-2.1-or-later")]).unwrap().len(), 1);
        assert_eq!(expression(&[c("MIT OR Apache-2.0"), c("MPL-2.0"), c("MPL-2.0")]), "(MIT OR Apache-2.0) AND MPL-2.0");
        assert_eq!(expression(&[c("MIT")]), "MIT");
    }

    #[test]
    fn globs_and_the_repository_toml() {
        assert!(glob_match("**", "a/b.viv") && glob_match("ports/dftatom/**", "ports/dftatom/states.viv"));
        assert!(glob_match("xc/*.json", "xc/catalog.json") && !glob_match("xc/*.json", "xc/tests/exact.json"));
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        assert_eq!(reuse_toml(&root.join("ports/dftatom/states.viv")).unwrap().0, "MIT");
        assert_eq!(reuse_toml(&root.join("lib/basis/basis.viv")).unwrap().0, "MIT OR Apache-2.0");
    }

    #[test]
    fn minpack_text_is_carried() {
        let cs = vec![Component { name: "lib/minpack/enorm.viv".into(), license: "LicenseRef-Minpack".into(), copyright: vec![] }];
        let n = notices(&cs, &|_| None);
        assert!(!n.contains("not in this compiler"), "{n}");
        assert!(n.contains("Minpack Copyright Notice"), "{n}");
        assert!(missing("LicenseRef-X").contains("LICENSES/") && !missing("LicenseRef-X").contains("spdx.org"));
    }
}
// REUSE-IgnoreEnd
