// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Compare an exact run's output with a fast run's, number by number.
//! `viv check --fast` uses it.
//!
//! The two outputs are split into lines, and each line into numbers and the
//! text between them. The text must be the same byte for byte: a test that
//! passes in one run passes in the other, a message is the same message, and
//! `nan` and `inf` are text, so a NaN or an infinity appears in both runs or
//! in neither. A number printed as an integer in both runs (an iteration
//! count, a size, an Int) must be the same integer: fast mode changes no
//! integer arithmetic and no control flow, so a different count is a failure,
//! not rounding. Every other number may differ by its tolerance:
//!
//! ```text
//! tol = max(rtol * max(|exact|, scale of the line), rtol * scale of the output,
//!           one unit in the last digit printed)
//!       (the finer of the two prints' last digits)
//! ```
//!
//! The line's scale is the largest magnitude printed on that line of the
//! exact output, so a small number computed as the difference of large ones
//! (a residual next to the energy it was taken from) is held to the error of
//! the large ones, not to its own size; and every number may differ by rtol
//! times the output's scale, the median magnitude of the exact run's nonzero
//! numbers (a drift "to rounding" printed on a line of its own). The report
//! counts the numbers that pass by that floor alone. The last printed digit covers a value
//! that sits on a rounding boundary of the format: a one-ulp change can move
//! a six-digit print by one unit. `rtol` is the run's budget, which the caller
//! states (section 4 of the design: an ulp budget per number, widened by the
//! condition numbers of the linear algebra the run did).

/// One number that differs between the runs.
#[derive(Debug, Clone)]
pub struct NumDiff {
    /// 1-based line and column in the exact output
    pub line: usize,
    pub col: usize,
    pub exact: String,
    pub fast: String,
    /// |fast - exact|
    pub abs: f64,
    /// |fast - exact| / |exact| (infinite for an exact zero)
    pub rel: f64,
    /// the difference in units of the last place of the exact value
    pub ulps: f64,
    pub tol: f64,
    /// within the tolerance
    pub ok: bool,
}

/// The comparison of two outputs.
#[derive(Debug, Default)]
pub struct Report {
    /// numbers compared
    pub numbers: usize,
    /// of them, the same text in both runs
    pub identical: usize,
    /// integers compared (printed as integers in both runs)
    pub integers: usize,
    /// every number that differs, in order
    pub diffs: Vec<NumDiff>,
    /// the output's scale (the median magnitude of its nonzero numbers)
    pub floor: f64,
    /// numbers within tolerance only by the output's scale
    pub floor_only: usize,
    /// differences that are not a number within tolerance: text, an integer,
    /// the line count, the shape of a line
    pub structural: Vec<String>,
}

impl Report {
    pub fn ok(&self) -> bool {
        self.structural.is_empty() && self.diffs.iter().all(|d| d.ok)
    }
    pub fn max_ulps(&self) -> f64 {
        self.diffs.iter().map(|d| d.ulps).filter(|u| u.is_finite()).fold(0.0, f64::max)
    }
    pub fn max_rel(&self) -> f64 {
        self.diffs.iter().map(|d| d.rel).filter(|r| r.is_finite()).fold(0.0, f64::max)
    }
    /// the largest |difference| / tolerance
    pub fn worst(&self) -> f64 {
        self.diffs.iter().map(|d| if d.tol > 0.0 { d.abs / d.tol } else { f64::INFINITY }).fold(0.0, f64::max)
    }
}

/// A piece of a line: text, or a number with its column.
#[derive(Debug, PartialEq)]
enum Piece<'a> {
    Text(&'a str),
    Num(&'a str, usize),
}

fn is_word(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// The length of the number starting at `i`, if one does: an optional sign,
/// digits with an optional fraction (or a fraction alone), an optional
/// exponent; not inside a word (`H2O`, `x1`, `6-31G` are text).
fn number_at(b: &[u8], i: usize) -> Option<usize> {
    if i > 0 && (is_word(b[i - 1]) || b[i - 1] == b'.') {
        return None;
    }
    let mut j = i;
    if j < b.len() && (b[j] == b'-' || b[j] == b'+') {
        j += 1;
    }
    let d0 = j;
    while j < b.len() && b[j].is_ascii_digit() {
        j += 1;
    }
    let mut digits = j - d0;
    if j < b.len() && b[j] == b'.' {
        let f0 = j + 1;
        let mut k = f0;
        while k < b.len() && b[k].is_ascii_digit() {
            k += 1;
        }
        digits += k - f0;
        if k - f0 > 0 || digits > 0 {
            j = k;
        }
    }
    if digits == 0 {
        return None;
    }
    if j < b.len() && (b[j] == b'e' || b[j] == b'E') {
        let mut k = j + 1;
        if k < b.len() && (b[k] == b'-' || b[k] == b'+') {
            k += 1;
        }
        let e0 = k;
        while k < b.len() && b[k].is_ascii_digit() {
            k += 1;
        }
        if k > e0 {
            j = k;
        }
    }
    if j < b.len() && (is_word(b[j]) || b[j] == b'.') {
        return None;
    }
    Some(j - i)
}

fn pieces(line: &str) -> Vec<Piece<'_>> {
    let b = line.as_bytes();
    let mut out = Vec::new();
    let (mut i, mut text0) = (0, 0);
    while i < b.len() {
        if let Some(n) = number_at(b, i) {
            if text0 < i {
                out.push(Piece::Text(&line[text0..i]));
            }
            out.push(Piece::Num(&line[i..i + n], i + 1));
            i += n;
            text0 = i;
        } else {
            i += 1;
        }
    }
    if text0 < b.len() {
        out.push(Piece::Text(&line[text0..]));
    }
    out
}

/// Printed as an integer: no point and no exponent, and small enough that a
/// double holds it exactly.
fn as_integer(s: &str) -> Option<i64> {
    if s.contains(['.', 'e', 'E']) {
        return None;
    }
    let v: i64 = s.parse().ok()?;
    (v.unsigned_abs() <= 1u64 << 53).then_some(v)
}

/// One unit in the last digit `s` prints: `0.125` -> 0.001, `1.5e-3` -> 1e-4.
fn last_digit_unit(s: &str) -> f64 {
    let (mant, exp) = match s.find(['e', 'E']) {
        Some(k) => (&s[..k], s[k + 1..].parse::<i32>().unwrap_or(0)),
        None => (s, 0),
    };
    let frac = mant.find('.').map(|k| mant.len() - k - 1).unwrap_or(0) as i32;
    10f64.powi(exp - frac)
}

/// The spacing of doubles at `x` (one ulp).
fn ulp(x: f64) -> f64 {
    let x = x.abs();
    if x == 0.0 || !x.is_finite() {
        return f64::MIN_POSITIVE * f64::EPSILON;
    }
    let next = f64::from_bits(x.to_bits() + 1);
    next - x
}

/// Compare the exact output with the fast one under the relative budget `rtol`.
pub fn compare(exact: &str, fast: &str, rtol: f64) -> Report {
    let mut r = Report::default();
    let (el, fl): (Vec<&str>, Vec<&str>) = (exact.lines().collect(), fast.lines().collect());
    if el.len() != fl.len() {
        r.structural.push(format!("the exact run printed {} lines, the fast run {}", el.len(), fl.len()));
    }
    // the output's scale: the median magnitude of the exact run's nonzero
    // numbers; a number may differ by rtol times it (a drift or a residual
    // "to rounding" printed on a line of its own)
    let mut mags: Vec<f64> = el
        .iter()
        .flat_map(|l| pieces(l))
        .filter_map(|p| match p {
            Piece::Num(s, _) => s.parse::<f64>().ok().map(f64::abs).filter(|v| *v > 0.0 && v.is_finite()),
            _ => None,
        })
        .collect();
    mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let floor = if mags.is_empty() { 0.0 } else { mags[mags.len() / 2] };
    r.floor = floor;
    for (k, (e, f)) in el.iter().zip(fl.iter()).enumerate() {
        let line = k + 1;
        let (ep, fp) = (pieces(e), pieces(f));
        let shape_same = ep.len() == fp.len()
            && ep.iter().zip(&fp).all(|(a, b)| match (a, b) {
                (Piece::Text(x), Piece::Text(y)) => x == y,
                (Piece::Num(..), Piece::Num(..)) => true,
                _ => false,
            });
        if !shape_same {
            r.structural.push(format!("line {line}: the text differs\n  exact: {}\n  fast:  {}", clip(e), clip(f)));
            continue;
        }
        let scale = ep
            .iter()
            .filter_map(|p| match p {
                Piece::Num(s, _) => s.parse::<f64>().ok().map(f64::abs),
                _ => None,
            })
            .fold(0.0, f64::max);
        for (a, b) in ep.iter().zip(&fp) {
            let (Piece::Num(es, col), Piece::Num(fs, _)) = (a, b) else { continue };
            r.numbers += 1;
            if es == fs {
                r.identical += 1;
                if as_integer(es).is_some() {
                    r.integers += 1;
                }
                continue;
            }
            if let (Some(x), Some(y)) = (as_integer(es), as_integer(fs)) {
                r.integers += 1;
                r.structural.push(format!("line {line}, column {col}: the integer {x} is {y} in the fast run"));
                continue;
            }
            let (x, y): (f64, f64) = (es.parse().unwrap_or(f64::NAN), fs.parse().unwrap_or(f64::NAN));
            let abs = (y - x).abs();
            // one unit of the last printed digit, with room for the error
            // of reading the decimal back
            let digit = last_digit_unit(es).min(last_digit_unit(fs)) * (1.0 + 1e-9);
            let rel = rtol * x.abs().max(scale);
            let tol = rel.max(digit).max(rtol * floor);
            let within_floor_only = abs > rel.max(digit) && abs <= rtol * floor;
            if within_floor_only {
                r.floor_only += 1;
            }
            r.diffs.push(NumDiff {
                line,
                col: *col,
                exact: es.to_string(),
                fast: fs.to_string(),
                abs,
                rel: if x == 0.0 { f64::INFINITY } else { abs / x.abs() },
                ulps: if x == 0.0 || !x.is_finite() { f64::INFINITY } else { abs / ulp(x) },
                tol,
                ok: abs <= tol,
            });
        }
    }
    r
}

fn clip(s: &str) -> String {
    if s.chars().count() > 150 { format!("{}...", s.chars().take(150).collect::<String>()) } else { s.to_string() }
}

/// The report as text: a summary line, then every number that differs (at
/// most `show`), the worst first.
pub fn render(r: &Report, show: usize) -> String {
    let mut out = String::new();
    let differ = r.diffs.len();
    let bad = r.diffs.iter().filter(|d| !d.ok).count();
    out.push_str(&format!(
        "{} numbers ({} printed as integers): {} identical, {} differ ({} beyond tolerance, {} within it only by the output's scale {:.3e}); largest difference {:.3e} relative, {:.1} ulps, {:.3} of its tolerance\n",
        r.numbers,
        r.integers,
        r.identical,
        differ,
        bad,
        r.floor_only,
        r.floor,
        r.max_rel(),
        r.max_ulps(),
        r.worst()
    ));
    for s in &r.structural {
        out.push_str(&format!("  DIFFERS: {s}\n"));
    }
    let mut order: Vec<&NumDiff> = r.diffs.iter().collect();
    order.sort_by(|a, b| (b.abs / b.tol).partial_cmp(&(a.abs / a.tol)).unwrap_or(std::cmp::Ordering::Equal));
    for d in order.iter().take(show) {
        out.push_str(&format!(
            "  {}line {}, col {}: exact {} fast {}  |d| {:.2e}  rel {:.2e}  {} ulps  {:.2e} of tolerance\n",
            if d.ok { "" } else { "BEYOND TOLERANCE " },
            d.line,
            d.col,
            d.exact,
            d.fast,
            d.abs,
            d.rel,
            if d.ulps.is_finite() { format!("{:.0}", d.ulps) } else { "-".to_string() },
            d.abs / d.tol
        ));
    }
    if order.len() > show {
        out.push_str(&format!("  ... and {} more\n", order.len() - show));
    }
    out
}

/// The report as one JSON object, for tools.
pub fn json(r: &Report) -> String {
    let num = |x: f64| if x.is_finite() { format!("{x:e}") } else { "null".to_string() };
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
    let diffs: Vec<String> = r
        .diffs
        .iter()
        .map(|d| {
            format!(
                "{{\"line\":{},\"col\":{},\"exact\":\"{}\",\"fast\":\"{}\",\"abs\":{},\"rel\":{},\"ulps\":{},\"tol\":{},\"ok\":{}}}",
                d.line,
                d.col,
                esc(&d.exact),
                esc(&d.fast),
                num(d.abs),
                num(d.rel),
                num(d.ulps),
                num(d.tol),
                d.ok
            )
        })
        .collect();
    let st: Vec<String> = r.structural.iter().map(|s| format!("\"{}\"", esc(s))).collect();
    format!(
        "{{\"ok\":{},\"numbers\":{},\"identical\":{},\"integers\":{},\"floor\":{},\"floor_only\":{},\"max_rel\":{},\"max_ulps\":{},\"worst\":{},\"structural\":[{}],\"diffs\":[{}]}}",
        r.ok(),
        r.numbers,
        r.identical,
        r.integers,
        num(r.floor),
        r.floor_only,
        num(r.max_rel()),
        num(r.max_ulps()),
        num(r.worst()),
        st.join(","),
        diffs.join(",")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_inside_words_are_text() {
        let p = pieces("H2O 6-31G* x1 = -1.5e-3, n 12");
        let nums: Vec<&str> = p.iter().filter_map(|x| if let Piece::Num(s, _) = x { Some(*s) } else { None }).collect();
        assert_eq!(nums, vec!["6", "-1.5e-3", "12"]);
    }

    #[test]
    fn identical_outputs_pass() {
        let r = compare("a 1.5\nPASS t\n", "a 1.5\nPASS t\n", 1e-12);
        assert!(r.ok());
        assert_eq!((r.numbers, r.identical), (1, 1));
    }

    #[test]
    fn a_last_bit_difference_is_within_tolerance() {
        let r = compare("e: -76.026765673119987\n", "e: -76.026765673119973\n", 1e-12);
        assert!(r.ok(), "{}", render(&r, 5));
        assert_eq!(r.diffs.len(), 1);
        assert!(r.diffs[0].ulps <= 2.0);
    }

    #[test]
    fn a_residual_is_held_to_its_lines_scale() {
        // 3e-15 vs 1e-15: far apart relatively, but the line's scale is 76
        assert!(compare("e -76.02 r 3e-15\n", "e -76.02 r 1e-15\n", 1e-12).ok());
        assert!(!compare("r 3e-15\n", "r 1e-15\n", 1e-12).ok());
    }

    #[test]
    fn integers_text_and_nan_must_match() {
        assert!(!compare("iterations 12\n", "iterations 13\n", 1.0).ok());
        assert!(!compare("PASS t\n", "FAIL t\n", 1.0).ok());
        assert!(!compare("x nan\n", "x 0.5\n", 1.0).ok());
        assert!(!compare("a\n", "a\nb\n", 1.0).ok());
        // an integral value printed short in one run is a number, not an integer
        assert!(compare("x 3\n", "x 2.9999999999999996\n", 1e-12).ok());
    }

    #[test]
    fn a_rounding_boundary_of_the_format_is_one_unit() {
        assert!(compare("x 0.123457\n", "x 0.123456\n", 1e-14).ok());
        assert!(!compare("x 0.123457\n", "x 0.123450\n", 1e-14).ok());
    }
}
