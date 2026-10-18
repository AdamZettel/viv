// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Text and numbers on one line: `print` with several arguments, `fmt`, and
//! `show a, b`.
//!
//! ```text
//! print("loss", loss)                  loss 0.693147         (%g, as `show`)
//! print("loss", fmt(loss))             loss 0.6931471805599453   (Python's repr)
//! print("loss", fmt(loss, 17))         loss 0.69314718055994529  (C's %.17g)
//! print("t =", fmt(t, ".3f"), "s")     t = 1.500 s           (a format spec)
//! show a, b                            a: 1  b: 2
//! ```
//!
//! `fmt(...)` is a string: a `print` argument, or a value
//! anywhere. `check` refuses a format that no value could take, with its
//! location, before any engine runs; the lowering refuses one the value's
//! type cannot.
//!
//! This file holds the formatting twice, line for line: in Rust, for the
//! interpreter and the wasm runtime (which calls the interpreter's code), and
//! in C (`C_RUNTIME`), for a compiled program. Both lean on the same two
//! correctly rounded primitives -- `%.*e`/`{:.*e}` and `strtod`/`parse` -- and
//! build everything else from them the same way, so the three engines print
//! the same bytes. `tools/print-fmt-sweep/` holds them to that over 10^6
//! doubles, and holds `fmt(x)` to CPython's `repr`.

use crate::ast::{Expr, Item, Program, Stmt};
use knotkit::diag::Diagnostic;
use knotkit::span::Span;

/// How one value is written.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// a bare `print` argument: `%g`, six significant digits, a negative
    /// zero written `0` -- what `show` and one-argument `print` write
    Bare,
    /// `fmt(x)`: the shortest decimal that reads back as x, laid out as
    /// Python's `repr(float)` lays it out
    Repr,
    /// `%.<prec>g`: `fmt(x, 17)`, `fmt(x, ".6g")`
    G,
    /// `%.<prec>e`: `fmt(x, ".3e")`
    E,
    /// `%.<prec>f`: `fmt(x, ".2f")`
    F,
    /// `fmt(x, "d")`: `%.0f`, for whole numbers
    D,
    /// `fmt(x, ".1%")`: 100 x in `%.<prec>f`, then `%`, as Python's `%`
    Pct,
    /// `str(x)` and `f"{x}"`: `fmt(x)`'s digits, but a whole number below
    /// 10^16 as its digits alone (`3`, not `3.0`), as `print` writes it
    /// (`str_num`). A spec with no type (`{x:>8}`) is Python's format, the
    /// repr.
    Str,
}

impl Kind {
    /// The letter the C runtime and the wasm bridge know it by.
    pub fn code(self) -> char {
        match self {
            Kind::Bare => 'b',
            Kind::Repr => 'r',
            Kind::G => 'g',
            Kind::E => 'e',
            Kind::F => 'f',
            Kind::D => 'd',
            Kind::Pct => '%',
            Kind::Str => 's',
        }
    }
    fn of_code(c: char) -> Option<Kind> {
        Some(match c {
            'b' => Kind::Bare,
            'r' => Kind::Repr,
            'g' => Kind::G,
            'e' => Kind::E,
            'f' => Kind::F,
            'd' => Kind::D,
            '%' => Kind::Pct,
            's' => Kind::Str,
            _ => return None,
        })
    }
}

/// A format: the kind, its precision, a minimum width (0 for none), and the
/// rest of a Python spec in `flags`: upper case (`E`, `F`, `G`), the sign
/// (`+`, ` `), the alignment (`<`, `>`, `^`, `=`), grouping (`,`, `_`) and
/// the fill character (`0` from the zero flag). A number is right-aligned and
/// space-filled when the flags say nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fmt {
    pub kind: Kind,
    pub prec: u32,
    pub width: u32,
    pub flags: u32,
}

/// The largest precision a spec may ask for, and the widest field.
pub const MAX_PREC: u32 = 40;
pub const MAX_WIDTH: u32 = 100;
/// A string format's largest precision (a number of characters).
pub const STR_MAX_PREC: u32 = 1 << 30;

// The bits of `Fmt::flags`. The C runtime and the wasm bridge get them above
// the kind's letter in one integer, the kind word (`Fmt::word`).
const FL_UPPER: u32 = 1;
const FL_SIGN_SHIFT: u32 = 1; // 2 bits: 0 `-`, 1 `+`, 2 ` `
const FL_ALIGN_SHIFT: u32 = 3; // 3 bits: 0 none, 1 `<`, 2 `>`, 3 `^`, 4 `=`
const FL_GROUP_SHIFT: u32 = 6; // 2 bits: 0 none, 1 `,`, 2 `_`
const FL_FILL_SHIFT: u32 = 8; // 7 bits: an ASCII fill, 0 for a space
/// The width (precision) is the value of a field nested in the spec,
/// `{x:>{w}}`, written `*` in the spec string the parser makes of it.
pub const FL_STAR_WIDTH: u32 = 1 << 15;
pub const FL_STAR_PREC: u32 = 1 << 16;
const FL_WORD: u32 = (1 << 15) - 1;

const SIGNS: [u8; 3] = [b'-', b'+', b' '];
const ALIGNS: [u8; 5] = [0, b'<', b'>', b'^', b'='];
const GROUPS: [u8; 3] = [0, b',', b'_'];

impl Fmt {
    pub const BARE: Fmt = Fmt { kind: Kind::Bare, prec: 6, width: 0, flags: 0 };
    pub const REPR: Fmt = Fmt { kind: Kind::Repr, prec: 0, width: 0, flags: 0 };
    pub const STR: Fmt = Fmt { kind: Kind::Str, prec: 0, width: 0, flags: 0 };

    /// `%.<digits>g`, the format `fmt(x, 17)` asks for.
    pub fn digits(d: u32) -> Fmt {
        Fmt { kind: Kind::G, prec: d, width: 0, flags: 0 }
    }

    /// How many of the width and precision are values (`*`).
    pub fn stars(&self) -> usize {
        (self.flags & FL_STAR_WIDTH != 0) as usize + (self.flags & FL_STAR_PREC != 0) as usize
    }
    pub fn upper(&self) -> bool {
        self.flags & FL_UPPER != 0
    }
    pub fn sign(&self) -> u8 {
        SIGNS[(self.flags >> FL_SIGN_SHIFT & 3) as usize % 3]
    }
    /// The alignment written (`<`, `>`, `^`, `=`), or 0 for a number's
    /// default, right.
    pub fn align(&self) -> u8 {
        ALIGNS[(self.flags >> FL_ALIGN_SHIFT & 7) as usize % 5]
    }
    pub fn group(&self) -> Option<u8> {
        Some(GROUPS[(self.flags >> FL_GROUP_SHIFT & 3) as usize % 3]).filter(|&g| g != 0)
    }
    pub fn fill(&self) -> u8 {
        match (self.flags >> FL_FILL_SHIFT & 127) as u8 {
            0 => b' ',
            c => c,
        }
    }

    /// The kind and the flags in one integer, as `viv_pnum` and the wasm
    /// bridge take them: the kind's letter in the low 8 bits.
    pub fn word(&self) -> u32 {
        self.kind.code() as u32 | (self.flags & FL_WORD) << 8
    }
    pub fn of_word(word: u32, prec: u32, width: u32) -> Option<Fmt> {
        let kind = Kind::of_code(char::from_u32(word & 255)?)?;
        Some(Fmt { kind, prec, width, flags: (word >> 8) & FL_WORD })
    }

    /// The format `fmt(x, ...)`'s arguments after x ask for: none (`fmt(x)`),
    /// a number of significant digits 1..=17 (`fmt(x, 17)`), or a spec string
    /// (`fmt(x, ".6f")`, or an f-string whose fields are its width and
    /// precision, `fmt(x, f".{p}f")`). The error names the argument at fault.
    pub fn of_args(args: &[Expr], call: Span) -> Result<Fmt, (Span, String, String)> {
        match args {
            [_] => Ok(Fmt::REPR),
            [_, Expr::Num { value, span }] => {
                if value.fract() == 0.0 && (1.0..=17.0).contains(value) {
                    Ok(Fmt::digits(*value as u32))
                } else {
                    Err((*span, "`fmt(x, digits)` takes 1 to 17 significant digits".into(), "a whole number from 1 to 17".into()))
                }
            }
            [_, Expr::Str { value, span }, rest @ ..] => {
                let f = Fmt::parse_spec(value).map_err(|m| (*span, m, "not a format this `fmt` knows".into()))?;
                if f.stars() == rest.len() {
                    Ok(f)
                } else if rest.is_empty() {
                    Err((*span, format!("format \"{value}\": `*` stands for a width or precision given as a value; write it in an f-string, f\".{{p}}f\""), "no value for the `*`".into()))
                } else {
                    Err((call, format!("`fmt` takes a value and at most one format, got {} arguments", args.len()), "wrong number of arguments".into()))
                }
            }
            [_, other] => Err((
                other.span(),
                "`fmt`'s second argument is a digit count or a format string, written in the source (an f-string may give its width and precision: f\".{p}f\")".into(),
                "write a number like 17 or a string like \".6f\" here".into(),
            )),
            _ => Err((call, format!("`fmt` takes a value and at most one format, got {} arguments", args.len()), "wrong number of arguments".into())),
        }
    }

    /// The formats `fmt(m, ...)` of a map asks for, keys then values: none
    /// (both `fmt(x)`), one (both in it), or two (the keys in the first, the
    /// values in the second). Only a map takes two;
    /// the engines refuse a second format on anything else.
    pub fn of_map_args(args: &[Expr], call: Span) -> Result<(Fmt, Fmt), (Span, String, String)> {
        let starred = matches!(args.get(1), Some(Expr::Str { value, .. }) if Fmt::parse_spec(value).is_ok_and(|f| f.stars() > 0));
        match args {
            [x, k, v] if !starred => Ok((Fmt::of_args(&[x.clone(), k.clone()], call)?, Fmt::of_args(&[x.clone(), v.clone()], call)?)),
            _ => Fmt::of_args(args, call).map(|f| (f, f)),
        }
    }

    /// A Python format spec, the part of an f-string field after the colon:
    /// `[[fill]align][sign][0][width][,|_][.precision][type]`, the type one
    /// of `e E f F g G d %` or none (`repr`). A width or precision written
    /// `*` comes from a nested field (`{x:>{w}}`). `#`, `z`, `n` and the
    /// integer types are refused, with what to write instead.
    pub fn parse_spec(s: &str) -> Result<Fmt, String> {
        let b = s.as_bytes();
        let mut i = 0;
        let mut flags = 0u32;
        let digits = |i: &mut usize| -> Option<u32> {
            let start = *i;
            while *i < b.len() && b[*i].is_ascii_digit() {
                *i += 1;
            }
            if *i == start {
                None
            } else {
                s[start..*i].parse().ok().or(Some(u32::MAX))
            }
        };
        let align_of = |c: u8| ALIGNS.iter().position(|&a| a != 0 && a == c);
        let mut fill = None;
        let mut align = None;
        if let Some(c) = s.chars().next() {
            let n = c.len_utf8();
            if let Some(a) = b.get(n).and_then(|&a| align_of(a)) {
                if !c.is_ascii() || c.is_ascii_control() {
                    return Err(format!("format \"{s}\": the fill is one ASCII character"));
                }
                fill = Some(c as u8);
                align = Some(a);
                i = n + 1;
            } else if let Some(a) = align_of(b[0]) {
                align = Some(a);
                i = 1;
            }
        }
        if let Some(k) = b.get(i).and_then(|c| SIGNS.iter().position(|s| s == c)) {
            flags |= (k as u32) << FL_SIGN_SHIFT;
            i += 1;
        }
        if let Some(c @ (b'#' | b'z')) = b.get(i) {
            return Err(format!("format \"{s}\": the `{}` flag is not supported", *c as char));
        }
        // the zero flag: zeros between the sign and the digits, Python's rule
        if b.get(i) == Some(&b'0') {
            fill = fill.or(Some(b'0'));
            align = align.or(Some(4));
            if b.get(i + 1) == Some(&b'*') {
                i += 1;
            }
        }
        let width = if b.get(i) == Some(&b'*') {
            i += 1;
            flags |= FL_STAR_WIDTH;
            0
        } else {
            digits(&mut i).unwrap_or(0)
        };
        if let Some(k) = b.get(i).and_then(|c| GROUPS.iter().position(|g| *g != 0 && g == c)) {
            flags |= (k as u32) << FL_GROUP_SHIFT;
            i += 1;
        }
        let mut prec = None;
        if i < b.len() && b[i] == b'.' {
            i += 1;
            if b.get(i) == Some(&b'*') {
                i += 1;
                flags |= FL_STAR_PREC;
                prec = Some(0);
            } else {
                prec = Some(digits(&mut i).ok_or_else(|| format!("format \"{s}\": a `.` must be followed by a precision"))?);
            }
        }
        let (kind, upper) = match &s[i..] {
            "" => (None, false),
            "e" => (Some(Kind::E), false),
            "E" => (Some(Kind::E), true),
            "f" => (Some(Kind::F), false),
            "F" => (Some(Kind::F), true),
            "g" => (Some(Kind::G), false),
            "G" => (Some(Kind::G), true),
            "d" => (Some(Kind::D), false),
            "%" => (Some(Kind::Pct), false),
            rest => {
                return Err(format!(
                    "format \"{s}\": `{rest}` is not supported; a format is [[fill]align][sign][0][width][,][.precision] then e, f, g, d or % (\".6g\", \"10.3f\", \"+.2e\", \">8\", \"08.3f\", \",d\")"
                ))
            }
        };
        if upper {
            flags |= FL_UPPER;
        }
        if width > MAX_WIDTH {
            return Err(format!("format \"{s}\": the width is at most {MAX_WIDTH}"));
        }
        if let Some(p) = prec {
            if p > MAX_PREC {
                return Err(format!("format \"{s}\": the precision is at most {MAX_PREC}"));
            }
        }
        if let Some(a) = align {
            flags |= (a as u32) << FL_ALIGN_SHIFT;
        }
        if let Some(f) = fill.filter(|&f| f != b' ') {
            flags |= (f as u32) << FL_FILL_SHIFT;
        }
        let kind = match (kind, prec) {
            (None, None) => Kind::Repr,
            (None, Some(_)) => {
                return Err(format!("format \"{s}\": a precision needs a type after it: write \"{s}g\" for significant digits or \"{s}f\" for places after the point"))
            }
            (Some(Kind::D), Some(_)) => return Err(format!("format \"{s}\": `d` takes no precision")),
            (Some(k), _) => k,
        };
        let prec = match kind {
            Kind::Repr | Kind::D | Kind::Bare | Kind::Str => 0,
            _ => prec.unwrap_or(6),
        };
        Ok(Fmt { kind, prec, width, flags })
    }

    /// The format as the wasm bridge carries it: `<kind><prec>.<width>`, and
    /// `.<flags>` after it when there are flags.
    pub fn encode(&self) -> String {
        let f = self.flags & FL_WORD;
        if f == 0 {
            format!("{}{}.{}", self.kind.code(), self.prec, self.width)
        } else {
            format!("{}{}.{}.{f}", self.kind.code(), self.prec, self.width)
        }
    }
    pub fn decode(s: &str) -> Option<Fmt> {
        let mut it = s.chars();
        let kind = Kind::of_code(it.next()?)?;
        let mut parts = it.as_str().split('.');
        let prec = parts.next()?.parse().ok()?;
        let width = parts.next()?.parse().ok()?;
        let flags = match parts.next() {
            Some(f) => f.parse().ok()?,
            None => 0,
        };
        Some(Fmt { kind, prec, width, flags })
    }
}

/// A format string with its width and precision as nested fields: the
/// f-string `f".{p}f"` (the parser's `viv__fstr(".", viv__str(p), "f")`, and
/// `{x:>{w}}`'s spec, which the parser writes the same way) is the spec
/// ".*f" and the expressions in it, in order. `None` for anything else.
pub fn star_spec(e: &Expr) -> Option<(String, Vec<Expr>)> {
    use crate::interp::strings::{FSTR, STR_OF};
    let Expr::Call { callee, args, .. } = e else { return None };
    if callee != FSTR {
        return None;
    }
    let mut text = String::new();
    let mut fields = Vec::new();
    for a in args {
        match a {
            Expr::Str { value, .. } => text.push_str(value),
            Expr::Call { callee, args, .. } if callee == STR_OF && args.len() == 1 => {
                text.push('*');
                fields.push(args[0].clone());
            }
            _ => return None,
        }
    }
    Some((text, fields))
}

impl Fmt {
    /// The format with its `*` width and precision given values: each must be
    /// a whole number, the width from 0 to `MAX_WIDTH`, the precision to
    /// `MAX_PREC`. Fields fill the width first, as they are written.
    pub fn with_stars(self, vals: &[f64]) -> Result<Fmt, String> {
        self.with_limits(vals, MAX_PREC)
    }
    /// `with_stars` with another largest precision: a string's precision is
    /// a count of characters, up to `STR_MAX_PREC`.
    pub fn with_limits(mut self, vals: &[f64], max_prec: u32) -> Result<Fmt, String> {
        let mut it = vals.iter();
        let mut take = |what: &str, max: u32| -> Result<u32, String> {
            let v = *it.next().ok_or_else(|| format!("a format's {what} field is missing"))?;
            if v.fract() != 0.0 || !(0.0..=max as f64).contains(&v) {
                return Err(format!("a format's {what} is {}; it must be a whole number from 0 to {max}", bare(v)));
            }
            Ok(v as u32)
        };
        if self.flags & FL_STAR_WIDTH != 0 {
            self.width = take("width", MAX_WIDTH)?;
        }
        if self.flags & FL_STAR_PREC != 0 {
            self.prec = take("precision", max_prec)?;
        }
        Ok(self)
    }
}

// ---- the numbers -----------------------------------------------------------

/// A number in a format. `nan` is `nan` whatever its sign bit, as Python
/// writes it (C would write `-nan` for some).
pub fn number(x: f64, f: Fmt) -> String {
    let s = match f.kind {
        Kind::Bare => bare(x),
        Kind::Repr => repr(x),
        Kind::Str => str_num(x),
        Kind::G if x.is_finite() => g(x, f.prec.max(1)),
        Kind::E if x.is_finite() => e(x, f.prec),
        Kind::F if x.is_finite() => format!("{:.*}", f.prec as usize, x),
        Kind::D if x.is_finite() => format!("{:.0}", x + 0.0),
        Kind::Pct if (x * 100.0).is_finite() => format!("{:.*}%", f.prec as usize, x * 100.0),
        Kind::Pct => format!("{}%", nonfinite(x * 100.0)),
        _ => nonfinite(x).to_string(),
    };
    finish(s, f)
}

/// An Int in a format that writes digits (bare, repr, `d`): its digits,
/// then the format's flags and width, as for a number.
pub fn int_text(i: i64, f: Fmt) -> String {
    finish(i.to_string(), f)
}

fn pad(s: String, width: u32) -> String {
    if s.len() >= width as usize {
        s
    } else {
        format!("{}{s}", " ".repeat(width as usize - s.len()))
    }
}

/// The digits `d` with a separator every three from the right, and zeros on
/// the left until there are at least `min` characters; never a separator
/// first (Python's zero padding with grouping: `f"{1234.5:010,.1f}"` is
/// `0,001,234.5`).
fn group(d: &[u8], sep: u8, min: usize) -> String {
    let mut out = Vec::with_capacity(d.len() + d.len() / 3 + min);
    let mut i = 0;
    while i < d.len() || out.len() < min {
        if i > 0 && i % 3 == 0 {
            out.push(sep);
        }
        out.push(if i < d.len() { d[d.len() - 1 - i] } else { b'0' });
        i += 1;
    }
    out.reverse();
    String::from_utf8(out).expect("ASCII")
}

/// A format's flags and width on a number's text: the sign the flags ask
/// for, upper case, grouping of the integer digits, then the fill on the
/// side the alignment says (`=` puts it between the sign and the digits).
/// `viv_numfin` in `C_RUNTIME` is this, line for line.
pub fn finish(s: String, f: Fmt) -> String {
    if f.flags & FL_WORD == 0 {
        return pad(s, f.width);
    }
    let (neg, body) = match s.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, s.as_str()),
    };
    let mut body = if f.upper() { body.to_ascii_uppercase() } else { body.to_string() };
    let sign = if neg {
        "-"
    } else {
        match f.sign() {
            b'+' => "+",
            b' ' => " ",
            _ => "",
        }
    };
    let width = f.width as usize;
    let (fill, align) = (f.fill(), f.align());
    if let Some(sep) = f.group() {
        let n = body.bytes().take_while(u8::is_ascii_digit).count();
        if n > 0 {
            let min = if fill == b'0' && align == b'=' { width.saturating_sub(sign.len() + body.len() - n) } else { 0 };
            body = group(&body.as_bytes()[..n], sep, min) + &body[n..];
        }
    }
    let len = sign.len() + body.len();
    if len >= width {
        return format!("{sign}{body}");
    }
    let k = width - len;
    let p = |m: usize| (fill as char).to_string().repeat(m);
    match align {
        b'<' => format!("{sign}{body}{}", p(k)),
        b'^' => format!("{}{sign}{body}{}", p(k / 2), p(k - k / 2)),
        b'=' => format!("{sign}{}{body}", p(k)),
        _ => format!("{}{sign}{body}", p(k)),
    }
}

fn nonfinite(x: f64) -> &'static str {
    if x.is_nan() {
        "nan"
    } else if x < 0.0 {
        "-inf"
    } else {
        "inf"
    }
}

/// Whether `x` is a whole number that `print` and `str` write as its
/// digits: below 10^16 in magnitude, where every digit `%.0f` writes is the
/// number's own (repr's positional range).
pub fn whole(x: f64) -> bool {
    x.fract() == 0.0 && x.abs() < 1e16
}

/// `%g`, as `show` writes a number, except a whole number below 10^16, which
/// is its digits (`1059490`, not `1.05949e+06`): a zero of either sign is `0`.
pub fn bare(x: f64) -> String {
    if x == 0.0 {
        "0".to_string()
    } else if !x.is_finite() {
        nonfinite(x).to_string()
    } else if whole(x) {
        format!("{x:.0}")
    } else {
        g(x, 6)
    }
}

/// `str(x)`: repr's shortest digits, but a whole number below 10^16 is its
/// digits alone, as `print` writes it (a zero of either sign `0`).
pub fn str_num(x: f64) -> String {
    if x == 0.0 {
        "0".to_string()
    } else if x.is_finite() && whole(x) {
        format!("{x:.0}")
    } else {
        repr(x)
    }
}

/// C's `%.<p>e`: Rust's `{:.p$e}` has the same digits but writes the
/// exponent `e5` where C writes `e+05`.
pub fn e(x: f64, p: u32) -> String {
    let s = format!("{:.*e}", p as usize, x);
    let (m, ex) = s.split_once('e').expect("{:e} has an exponent");
    let ex: i32 = ex.parse().expect("a decimal exponent");
    format!("{m}e{}{:02}", if ex < 0 { '-' } else { '+' }, ex.abs())
}

/// C's `%.<p>g`, p >= 1, x finite: the exponent that picks the form is the
/// one `%.<p-1>e` writes after rounding, as C specifies; trailing zeros go.
/// `viv_fmt_g` in `C_RUNTIME` is this, built the same way from `%e` and `%f`
/// rather than taken from the C library's `%g` (see there).
pub fn g(x: f64, p: u32) -> String {
    let sci = format!("{:.*e}", p as usize - 1, x);
    let exp: i32 = sci.split_once('e').and_then(|(_, e)| e.parse().ok()).unwrap_or(0);
    if exp < -4 || exp >= p as i32 {
        let (m, _) = sci.split_once('e').expect("{:e} has an exponent");
        format!("{}e{}{:02}", trim_zeros(m), if exp < 0 { '-' } else { '+' }, exp.abs())
    } else {
        trim_zeros(&format!("{:.*}", (p as i32 - 1 - exp) as usize, x)).to_string()
    }
}

fn trim_zeros(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        s
    }
}

/// The shortest decimal that reads back as |x| (x finite, nonzero): its
/// digits and the exponent of the first. For each length p from 1, the
/// candidates are the p-digit decimal nearest x (`%.<p-1>e`, correctly
/// rounded) and the p-digit decimal on x's other side, which is the one that
/// reads back when x is a power of two (the gap below it is half the gap
/// above). Of those that read back, the nearer wins, which is the digits
/// Python's `repr` and Rust's `{}` choose. `viv_repr` in `C_RUNTIME` is this,
/// line for line.
fn shortest(a: f64) -> (String, i32) {
    for p in 1..=17u32 {
        let s = format!("{:.*e}", p as usize - 1, a);
        let (m, ex) = s.split_once('e').expect("{:e} has an exponent");
        let mut n: u64 = m.bytes().filter(|c| c.is_ascii_digit()).fold(0, |n, c| n * 10 + (c - b'0') as u64);
        let mut ex: i32 = ex.parse().expect("a decimal exponent");
        let r: f64 = s.parse().expect("a decimal");
        if r != a {
            let lo = 10u64.pow(p - 1);
            let hi = lo * 10;
            if r < a {
                n += 1;
                if n == hi {
                    n = lo;
                    ex += 1;
                }
            } else {
                n -= 1;
                if n < lo {
                    n = hi - 1;
                    ex -= 1;
                }
            }
            let t = format!("{}e{}", n, ex - (p as i32 - 1));
            if t.parse::<f64>() != Ok(a) {
                continue;
            }
        }
        let d = n.to_string();
        let d = d.trim_end_matches('0');
        return (if d.is_empty() { "0".into() } else { d.to_string() }, ex);
    }
    unreachable!("17 significant digits always read back")
}

/// `fmt(x)`: Python's `repr(float)`, exactly. The shortest round-trip
/// digits; positional notation when the decimal point falls within 16 places
/// of them (1e-4 <= |x| < 1e16), with `.0` on a whole number; otherwise
/// `d.ddde+XX`. `-0.0`, `inf`, `-inf`, `nan` as Python writes them.
pub fn repr(x: f64) -> String {
    if !x.is_finite() {
        return nonfinite(x).to_string();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    let (d, ex) = shortest(x.abs());
    repr_layout(x < 0.0, &d, ex)
}

/// `repr`'s layout of the shortest digits `d` (no trailing zeros) whose
/// first digit has decimal exponent `ex`: also how a Float32 is written
/// (interp/f32s.rs), with its own shortest digits.
pub fn repr_layout(neg: bool, d: &str, ex: i32) -> String {
    let mut s = String::with_capacity(26);
    if neg {
        s.push('-');
    }
    let nd = d.len() as i32;
    let decpt = ex + 1;
    if -4 < decpt && decpt <= 16 {
        if decpt <= 0 {
            s.push_str("0.");
            for _ in 0..-decpt {
                s.push('0');
            }
            s.push_str(d);
        } else if decpt >= nd {
            s.push_str(d);
            for _ in 0..decpt - nd {
                s.push('0');
            }
            s.push_str(".0");
        } else {
            s.push_str(&d[..decpt as usize]);
            s.push('.');
            s.push_str(&d[decpt as usize..]);
        }
    } else {
        s.push_str(&d[..1]);
        if nd > 1 {
            s.push('.');
            s.push_str(&d[1..]);
        }
        s.push_str(&format!("e{}{:02}", if ex < 0 { '-' } else { '+' }, ex.abs()));
    }
    s
}

/// An array, one line, nested by its shape: `[1, 2]`, `[[1, 2], [3, 4]]`.
pub fn array(shape: &[usize], data: &[f64], f: Fmt) -> String {
    let mut s = String::new();
    nest(shape, data, f, &mut s);
    s
}

fn nest(shape: &[usize], data: &[f64], f: Fmt, s: &mut String) {
    s.push('[');
    if shape.len() <= 1 {
        for (i, &x) in data.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(&number(x, f));
        }
    } else {
        let block: usize = shape[1..].iter().product();
        for i in 0..shape[0] {
            if i > 0 {
                s.push_str(", ");
            }
            nest(&shape[1..], &data[i * block..(i + 1) * block], f, s);
        }
    }
    s.push(']');
}

// ---- the C twin --------------------------------------------------------------

/// The C runtime's half: `viv_print(n, ...)` writes one line from `n` parts,
/// each a tag and its payload -- 0: a `const char*` text; 1: a number
/// (`double`, then kind, precision, width as `int`s); 2: a `viv_arr` and the
/// same three `int`s; 3: a boolean (`double`); 4: a complex number
/// (`_Complex double`, which C lays out as two doubles). Every function is
/// the Rust function of the same name above, line for line.
/// C's `%g` and `%.<p>g` for a compiled program, built by `viv_fmt_g` the
/// way `g` is built, never by the C library's `%g`. `viv_g` is `bare` (what
/// `show`, one-argument `print`, the charts' short arrays and the assert
/// messages write), `viv_gp` is `g` with nan and the infinities as `bare`
/// writes them. Each returns one of eight rotating buffers, so one printf can
/// take several. The runtime fragment "fmt_g" (lower.rs).
pub const C_FMT_G: &str = r#"/* C's %g without the C library's %g (printfmt.rs, line for line) */
/* %.<p>g from %.<p-1>e and %.<p>f, not from the C library's %g: macOS's
   viv_printf("%g", 7890305.0) writes 7.89030e+06, keeping a trailing zero when the
   rounding is a tie, where C (and every other engine) says 7.8903e+06 */
static void viv_trim(char* m) {
    if (!strchr(m, '.')) return;
    size_t n = strlen(m);
    while (m[n - 1] == '0') m[--n] = 0;
    if (m[n - 1] == '.') m[--n] = 0;
}
/* the exponent %e wrote: atoi without <stdlib.h>, whose declarations (wait,
   exit, abs, ...) a program's own defs may be named like */
static int viv_exp10(const char* c) {
    int neg = *c == '-', v = 0;
    if (*c == '-' || *c == '+') c++;
    while (*c >= '0' && *c <= '9') v = v * 10 + (*c++ - '0');
    return neg ? -v : v;
}
static void viv_fmt_g(char* b, int sz, double x, int p) {
    char s[64];
    snprintf(s, sizeof s, "%.*e", p - 1, x);
    char* e = strchr(s, 'e');
    int ex = viv_exp10(e + 1);
    if (ex < -4 || ex >= p) {
        *e = 0;
        viv_trim(s);
        snprintf(b, sz, "%se%c%02d", s, ex < 0 ? '-' : '+', ex < 0 ? -ex : ex);
    } else {
        snprintf(b, sz, "%.*f", p - 1 - ex, x);
        viv_trim(b);
    }
}
static const char* viv_gp(double x, int p) {
    static char bufs[8][64];
    static int k = 0;
    char* b = bufs[k = (k + 1) & 7];
    if (x != x) return "nan";
    if (x > DBL_MAX || x < -DBL_MAX) return x < 0 ? "-inf" : "inf";
    viv_fmt_g(b, 64, x, p < 1 ? 1 : p);
    return b;
}
/* a whole number below 10^16, which print and str write as its digits
   (printfmt.rs, whole) */
static int viv_whole(double x) { return x < 1e16 && x > -1e16 && x == (double)(long long)x; }
static const char* viv_g(double x) {
    static char bufs[8][32];
    static int k = 0;
    if (x == 0.0) return "0";
    if (viv_whole(x)) {
        char* b = bufs[k = (k + 1) & 7];
        snprintf(b, sizeof bufs[0], "%.0f", x);
        return b;
    }
    return viv_gp(x, 6);
}
/* a number at full precision: the shortest of %.15g, %.16g, %.17g that reads
   back as the same double (the interpreter's format_rt); how an index that is
   not an integer is shown in its error */
static void viv_rt_text(char* b, int sz, double x) {
    x += 0.0;
    if (x != x) { snprintf(b, sz, "nan"); return; }
    if (x > DBL_MAX || x < -DBL_MAX) { snprintf(b, sz, x < 0 ? "-inf" : "inf"); return; }
    if (x == 0.0) { snprintf(b, sz, "0"); return; }
    for (int p = 15; p <= 16; p++) {
        double y;
        viv_fmt_g(b, sz, x, p);
        if (sscanf(b, "%lf", &y) == 1 && y == x) return;
    }
    viv_fmt_g(b, sz, x, 17);
}"#;

pub const C_RUNTIME: &str = r#"/* print with text and numbers, and fmt (printfmt.rs, line for line) */
static void viv_fmt_e(char* b, int sz, double x, int p) { snprintf(b, sz, "%.*e", p, x); }
static void viv_repr(char* out, double x) {
    if (x != x) { strcpy(out, "nan"); return; }
    if (x > DBL_MAX || x < -DBL_MAX) { strcpy(out, x < 0 ? "-inf" : "inf"); return; }
    if (x == 0.0) { strcpy(out, signbit(x) ? "-0.0" : "0.0"); return; }
    double a = x < 0 ? -x : x;
    char b[48], t[48], d[24];
    unsigned long long n = 0;
    int ex = 0, p;
    for (p = 1; p <= 17; p++) {
        snprintf(b, sizeof b, "%.*e", p - 1, a);
        const char* c = b;
        n = 0;
        for (; *c != 'e'; c++) if (*c >= '0' && *c <= '9') n = n * 10 + (unsigned long long)(*c - '0');
        ex = atoi(c + 1);
        double r = strtod(b, 0);
        if (r != a) {
            unsigned long long lo = 1, hi;
            for (int k = 1; k < p; k++) lo *= 10;
            hi = lo * 10;
            if (r < a) { n++; if (n == hi) { n = lo; ex++; } }
            else { n--; if (n < lo) { n = hi - 1; ex--; } }
            snprintf(t, sizeof t, "%llue%d", n, ex - (p - 1));
            if (strtod(t, 0) != a) continue;
        }
        break;
    }
    snprintf(d, sizeof d, "%llu", n);
    int nd = (int)strlen(d);
    while (nd > 1 && d[nd - 1] == '0') nd--;
    d[nd] = 0;
    char* s = out;
    if (x < 0) *s++ = '-';
    int decpt = ex + 1;
    if (-4 < decpt && decpt <= 16) {
        if (decpt <= 0) {
            *s++ = '0'; *s++ = '.';
            for (int k = 0; k < -decpt; k++) *s++ = '0';
            strcpy(s, d);
        } else if (decpt >= nd) {
            strcpy(s, d); s += nd;
            for (int k = 0; k < decpt - nd; k++) *s++ = '0';
            strcpy(s, ".0");
        } else {
            memcpy(s, d, (size_t)decpt); s += decpt;
            *s++ = '.';
            strcpy(s, d + decpt);
        }
    } else {
        *s++ = d[0];
        if (nd > 1) { *s++ = '.'; strcpy(s, d + 1); s += nd - 1; }
        sprintf(s, "e%c%02d", ex < 0 ? '-' : '+', ex < 0 ? -ex : ex);
    }
}
/* a format's flags and width on a number's text s, into out (printfmt.rs
   `finish` and `group`, line for line). word is the kind word: the kind's
   letter, then the flags from bit 8 -- bit 0 upper case, 1-2 the sign
   (- + space), 3-5 the alignment (none < > ^ =), 6-7 grouping (none , _),
   8-14 the fill (0 for a space). out holds 1024 bytes: s is at most 360
   (%.40f of 1e308), its grouping a third more, the width 100. */
static void viv_numfin(char* out, const char* s, int word, int width) {
    int fl = (word >> 8) & 32767;
    int n = (int)strlen(s);
    if (fl == 0) {
        int k = 0;
        for (; k < width - n; k++) out[k] = ' ';
        memcpy(out + k, s, (size_t)n + 1);
        return;
    }
    int neg = s[0] == '-';
    if (neg) { s++; n--; }
    char body[600];
    for (int i = 0; i <= n; i++) body[i] = (fl & 1) && s[i] >= 'a' && s[i] <= 'z' ? (char)(s[i] - 32) : s[i];
    int sg = (fl >> 1) & 3, al = (fl >> 3) & 7, gr = (fl >> 6) & 3, fc = (fl >> 8) & 127;
    const char* sign = neg ? "-" : sg == 1 ? "+" : sg == 2 ? " " : "";
    int sl = (int)strlen(sign);
    char fill = fc ? (char)fc : ' ';
    char align = al == 1 ? '<' : al == 2 ? '>' : al == 3 ? '^' : al == 4 ? '=' : 0;
    if (gr == 1 || gr == 2) {
        char sep = gr == 1 ? ',' : '_';
        int nd = 0;
        while (body[nd] >= '0' && body[nd] <= '9') nd++;
        if (nd > 0) {
            int min = 0;
            if (fill == '0' && align == '=') { min = width - (sl + n - nd); if (min < 0) min = 0; }
            char g[600];
            int m = 0, i = 0;
            while (i < nd || m < min) {
                if (i > 0 && i % 3 == 0) g[m++] = sep;
                g[m++] = i < nd ? body[nd - 1 - i] : '0';
                i++;
            }
            char t[600];
            for (int k = 0; k < m; k++) t[k] = g[m - 1 - k];
            memcpy(t + m, body + nd, (size_t)(n - nd) + 1);
            n = m + n - nd;
            memcpy(body, t, (size_t)n + 1);
        }
    }
    int len = sl + n, k = width - len, o = 0;
    if (k <= 0) { memcpy(out, sign, (size_t)sl); memcpy(out + sl, body, (size_t)n + 1); return; }
    int left = align == '<' ? 0 : align == '^' ? k / 2 : align == '=' ? 0 : k;
    int mid = align == '=' ? k : 0;
    for (int q = 0; q < left; q++) out[o++] = fill;
    memcpy(out + o, sign, (size_t)sl); o += sl;
    for (int q = 0; q < mid; q++) out[o++] = fill;
    memcpy(out + o, body, (size_t)n); o += n;
    for (int q = left + mid; q < k; q++) out[o++] = fill;
    out[o] = 0;
}
/* a number's text in a format, before the flags: kind 'b' bare %g, 'r'
   repr, 'g' 'e' 'f' 'd', '%' (100 x in %f, then %) */
static void viv_numtext(char* b, int sz, double x, int kind, int prec) {
    if ((kind == 's' || kind == 'b') && x == 0.0) strcpy(b, "0");
    else if ((kind == 's' || kind == 'b') && viv_whole(x)) snprintf(b, (size_t)sz, "%.0f", x);
    else if (kind == 'r' || kind == 's') viv_repr(b, x);
    else if (kind == '%') {
        double y = x * 100.0;
        if (y != y) strcpy(b, "nan%");
        else if (y > DBL_MAX || y < -DBL_MAX) strcpy(b, y < 0 ? "-inf%" : "inf%");
        else { snprintf(b, (size_t)sz - 1, "%.*f", prec, y); strcat(b, "%"); }
    }
    else if (kind == 'b' && x == 0.0) strcpy(b, "0");
    else if (x != x) strcpy(b, "nan");
    else if (x > DBL_MAX || x < -DBL_MAX) strcpy(b, x < 0 ? "-inf" : "inf");
    else if (kind == 'b') viv_fmt_g(b, sz, x, 6);
    else if (kind == 'g') viv_fmt_g(b, sz, x, prec < 1 ? 1 : prec);
    else if (kind == 'e') viv_fmt_e(b, sz, x, prec);
    else if (kind == 'f') snprintf(b, (size_t)sz, "%.*f", prec, x);
    else snprintf(b, (size_t)sz, "%.0f", x + 0.0); /* d: a negative zero is 0 */
}
/* one number in a format (the kind word: see viv_numfin) */
static void viv_pnum(double x, int kind, int prec, int width) {
    static char b[512], o[1024];
    viv_numtext(b, sizeof b, x, kind & 255, prec);
    viv_numfin(o, b, kind, width);
    viv_fputs(o, stdout);
}
static void viv_parr(const double* data, const int* dims, int rank, int kind, int prec, int width) {
    viv_putchar('[');
    if (rank <= 1) {
        for (int i = 0; i < dims[0]; i++) { if (i) viv_fputs(", ", stdout); viv_pnum(data[i], kind, prec, width); }
    } else {
        int block = 1;
        for (int k = 1; k < rank; k++) block *= dims[k];
        for (int i = 0; i < dims[0]; i++) { if (i) viv_fputs(", ", stdout); viv_parr(data + (size_t)i * block, dims + 1, rank - 1, kind, prec, width); }
    }
    viv_putchar(']');
}
static double viv_print(int n, ...) {
    va_list ap;
    va_start(ap, n);
    for (int k = 0; k < n; k++) {
        int tag = va_arg(ap, int);
        if (tag == 0) viv_fputs(va_arg(ap, const char*), stdout);
        else if (tag == 1) { double x = va_arg(ap, double); int kd = va_arg(ap, int), p = va_arg(ap, int), w = va_arg(ap, int); viv_pnum(x, kd, p, w); }
        else if (tag == 2) {
            viv_arr a = va_arg(ap, viv_arr);
            int kd = va_arg(ap, int), p = va_arg(ap, int), w = va_arg(ap, int);
            int one = a.size;
            viv_parr(a.data, a.rank <= 1 ? &one : a.dims, a.rank, kd, p, w);
        }
        else if (tag == 3) viv_fputs(va_arg(ap, double) != 0.0 ? "true" : "false", stdout);
        else {
            _Complex double z = va_arg(ap, _Complex double);
            double re = ((double*)&z)[0] + 0.0, im = ((double*)&z)[1] + 0.0;
            viv_pnum(re, 'b', 6, 0);
            viv_fputs(im < 0.0 ? "-" : "+", stdout);
            viv_pnum(im < 0.0 ? -im : im, 'b', 6, 0);
            viv_putchar('i');
        }
    }
    va_end(ap);
    viv_putchar('\n');
    return 0.0;
}"#;

// ---- the static check ------------------------------------------------------

/// Refuse, before any engine runs, a `fmt` that is not a `print` argument and
/// a `fmt` whose format is not one this file knows. A program's own `def fmt`
/// or `def print` is its function, and is left alone.
pub fn check(program: &Program) -> Result<(), Vec<Diagnostic>> {
    let own = |name: &str| program.items.iter().any(|i| matches!(i, Item::Def(d) if d.name == name));
    let cx = Cx { fmt: !own("fmt"), print: !own("print") };
    let mut errs = Vec::new();
    for item in &program.items {
        match item {
            Item::Def(d) => cx.stmts(&d.body, &mut errs),
            Item::Stmt(s) => cx.stmts(std::slice::from_ref(s), &mut errs),
            _ => {}
        }
    }
    if errs.is_empty() {
        Ok(())
    } else {
        Err(errs)
    }
}

struct Cx {
    /// `fmt` and `print` are the builtins (the program defines neither)
    fmt: bool,
    print: bool,
}

impl Cx {
    fn stmts(&self, body: &[Stmt], errs: &mut Vec<Diagnostic>) {
        for s in body {
            match s {
                Stmt::Assign { value: e, .. } | Stmt::Return { value: e, .. } | Stmt::ExprStmt { expr: e, .. } | Stmt::Show { expr: e, .. } => self.expr(e, errs),
                Stmt::Test { body, .. } => self.stmts(body, errs),
                Stmt::SetIndex { base, indices, value, .. } => {
                    self.expr(base, errs);
                    for i in indices {
                        self.expr(i, errs);
                    }
                    self.expr(value, errs);
                }
                Stmt::SetField { base, value, .. } => {
                    self.expr(base, errs);
                    self.expr(value, errs);
                }
                Stmt::If { cond, then_body, else_body, .. } => {
                    self.expr(cond, errs);
                    self.stmts(then_body, errs);
                    self.stmts(else_body, errs);
                }
                Stmt::While { cond, body, .. } => {
                    self.expr(cond, errs);
                    self.stmts(body, errs);
                }
                Stmt::For { bound, body, .. } => {
                    self.expr(bound, errs);
                    self.stmts(body, errs);
                }
                _ => {}
            }
        }
    }

    fn expr(&self, e: &Expr, errs: &mut Vec<Diagnostic>) {
        match e {
            Expr::Call { callee, args, .. } if callee == "print" && self.print => {
                for a in args {
                    match a {
                        // a map's `fmt` in a print takes one or two formats
                        // a string's, its own
                        Expr::Call { callee, args: fa, span } if callee == "fmt" && self.fmt => {
                            let string_spec = matches!(fa.as_slice(), [_, Expr::Str { value, .. }, ..] if crate::interp::strings::StrSpec::parse(value).is_ok());
                            if !string_spec {
                                if let Err((at, msg, label)) = Fmt::of_map_args(fa, *span) {
                                    errs.push(Diagnostic::error(msg).with_primary(at, label));
                                }
                            }
                            for x in fa {
                                self.expr(x, errs);
                            }
                        }
                        a => self.expr(a, errs),
                    }
                }
            }
            // `fmt` is a string anywhere; its format is
            // checked here when it can be neither a number's nor a string's,
            // and by the lowering, which knows which the value is
            Expr::Call { callee, args: fa, span } if callee == "fmt" && self.fmt => {
                let string_spec = matches!(fa.as_slice(), [_, Expr::Str { value, .. }, ..] if crate::interp::strings::StrSpec::parse(value).is_ok());
                if !string_spec {
                    if let Err((at, msg, label)) = Fmt::of_args(fa, *span) {
                        errs.push(Diagnostic::error(msg).with_primary(at, label));
                    }
                }
                for a in fa {
                    self.expr(a, errs);
                }
            }
            Expr::Call { args, .. } => {
                for a in args {
                    self.expr(a, errs);
                }
            }
            Expr::ArrLit { elems, .. } => {
                for a in elems {
                    self.expr(a, errs);
                }
            }
            Expr::Index { base, indices, .. } => {
                self.expr(base, errs);
                for i in indices {
                    self.expr(i, errs);
                }
            }
            Expr::Lambda { body, .. } => self.expr(body, errs),
            Expr::Neg { operand, .. } | Expr::Not { operand, .. } => self.expr(operand, errs),
            Expr::Binary { lhs, rhs, .. } => {
                self.expr(lhs, errs);
                self.expr(rhs, errs);
            }
            Expr::RecordLit { fields, .. } => {
                for (_, _, v) in fields {
                    self.expr(v, errs);
                }
            }
            Expr::Field { base, .. } => self.expr(base, errs),
            Expr::Imag { .. } | Expr::Bool { .. } | Expr::Num { .. } | Expr::Str { .. } | Expr::Var { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr_matches_python_on_known_values() {
        for (x, want) in [
            (0.1, "0.1"),
            (1.0, "1.0"),
            (-0.0, "-0.0"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (1.5e-5, "1.5e-05"),
            (123456789012345678.0, "1.2345678901234568e+17"),
            (5e-324, "5e-324"),
            (2.2250738585072014e-308, "2.2250738585072014e-308"),
            (f64::MAX, "1.7976931348623157e+308"),
            (0.3, "0.3"),
            (0.1 + 0.2, "0.30000000000000004"),
            (-20195.654797017647, "-20195.654797017647"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
            (f64::NAN, "nan"),
        ] {
            assert_eq!(repr(x), want, "repr({x:e})");
        }
    }

    #[test]
    fn repr_digits_are_as_short_as_rusts() {
        // Rust's `{:e}` is shortest-round-trip too, so the lengths must agree.
        // The digits may not: when two shortest decimals are equally near,
        // Rust takes the upper and Python (and this) the even one --
        // 1658206780088562.25 is ...562.2 here and in Python, ...562.3 in Rust.
        let mut s = 0x9E3779B97F4A7C15u64;
        for _ in 0..200_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let x = f64::from_bits(s);
            if !x.is_finite() || x == 0.0 {
                continue;
            }
            let (d, ex) = shortest(x.abs());
            let want = format!("{:e}", x.abs());
            let (m, we) = want.split_once('e').unwrap();
            assert_eq!(d.len(), m.replace('.', "").len(), "{want}");
            assert_eq!(format!("{d}e{}", ex - (d.len() as i32 - 1)).parse::<f64>(), Ok(x.abs()), "{want}");
        }
    }

    #[test]
    fn specs() {
        assert_eq!(Fmt::parse_spec(".6g"), Ok(Fmt { kind: Kind::G, prec: 6, width: 0, flags: 0 }));
        assert_eq!(Fmt::parse_spec("10.3f"), Ok(Fmt { kind: Kind::F, prec: 3, width: 10, flags: 0 }));
        assert_eq!(Fmt::parse_spec("e"), Ok(Fmt { kind: Kind::E, prec: 6, width: 0, flags: 0 }));
        assert_eq!(Fmt::parse_spec(""), Ok(Fmt::REPR));
        assert_eq!(Fmt::parse_spec("8"), Ok(Fmt { kind: Kind::Repr, prec: 0, width: 8, flags: 0 }));
        assert_eq!(Fmt::parse_spec("5d"), Ok(Fmt { kind: Kind::D, prec: 0, width: 5, flags: 0 }));
        for bad in ["#x", "z.3f", ".3", ".3d", "x", ".f", ".41f", "101g", "n", "\u{e9}<5"] {
            assert!(Fmt::parse_spec(bad).is_err(), "{bad}");
        }
        let f = Fmt { kind: Kind::F, prec: 3, width: 10, flags: 0 };
        assert_eq!(Fmt::decode(&f.encode()), Some(f));
    }

    #[test]
    fn numbers() {
        assert_eq!(number(0.5, Fmt::parse_spec(".3f").unwrap()), "0.500");
        assert_eq!(number(0.125, Fmt::parse_spec(".2f").unwrap()), "0.12");
        assert_eq!(number(123456.0, Fmt::parse_spec(".2e").unwrap()), "1.23e+05");
        assert_eq!(number(999999.5, Fmt::BARE), "1e+06");
        assert_eq!(number(-0.0, Fmt::BARE), "0");
        assert_eq!(number(-0.0, Fmt::parse_spec(".3g").unwrap()), "-0");
        assert_eq!(number(f64::NAN, Fmt::parse_spec("6.2f").unwrap()), "   nan");
        assert_eq!(number(0.1, Fmt::digits(17)), "0.10000000000000001");
        assert_eq!(number(2.5, Fmt::parse_spec("d").unwrap()), "2");
        assert_eq!(array(&[2, 2], &[1.0, 2.0, 3.0, 4.5], Fmt::BARE), "[[1, 2], [3, 4.5]]");
        assert_eq!(array(&[0], &[], Fmt::BARE), "[]");
    }
}
