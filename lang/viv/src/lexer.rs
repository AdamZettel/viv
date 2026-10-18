// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Viv's lexer, aligned with the real Viv surface: `#` comments, `Newline`
//! tokens as statement terminators (suppressed inside `(...)` for implicit line
//! continuation), `def`/`return` keywords, `{}` blocks, and a single numeric
//! type (all numbers lex to `f64`). Lex errors surface as diagnostics (1.7).

use knotkit::diag::Diagnostic;
use knotkit::span::{SourceId, Span};

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    /// An imaginary literal: `2.5i` is 2.5 times the imaginary unit.
    Imag(f64),
    Def,
    Return,
    If,
    Else,
    While,
    For,
    To,
    Break,
    Continue,
    Narrate,
    Show,
    Test,
    Rung,
    Parallel,
    Fn,
    Record,
    Arrow,
    Colon,
    Dot,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Eq,
    Plus,
    Minus,
    Star,
    Slash,
    At,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    /// `%`: floored modulo.
    Percent,
    /// `//`: floor division.
    SlashSlash,
    /// `//=`
    SlashSlashEq,
    /// `^`: power, `x ^ y` is `pow(x, y)`.
    Caret,
    /// `=>`: a `match` arm's pattern from its body.
    FatArrow,
    /// `?`: an optional type, `Int?`.
    Question,
    /// `|`: one `match` arm for several variants, `A | B => ...`.
    Pipe,
    Str(String),
    /// An f-string, `f"x = {x:.3f}"`: its text and its fields, in order.
    FStr(Vec<FPart>),
    // Comparisons.
    Lt,
    Le,
    Gt,
    Ge,
    EqEq,
    Ne,
    // Logical (word forms `and`/`or`/`not` lex to these too).
    And,
    Or,
    Not,
    /// `true` / `false`: the two values of `Bool`.
    True,
    False,
    Ident(String),
    /// A number literal, and, when it was written as an integer (digits with
    /// no `.` and no exponent: `3`, not `3.0` or `3e0`) that fits in 64 bits,
    /// its exact value.
    Num(f64, Option<i64>),
    /// Statement terminator (newline or `;`), suppressed inside parentheses.
    Newline,
    Eof,
}

/// One piece of an f-string.
#[derive(Clone, Debug, PartialEq)]
pub enum FPart {
    /// text, escapes decoded (`{{` and `}}` are single braces)
    Text(String),
    /// `{expr[=][!c][:spec]}`: the expression's tokens (ending in `Eof`); the
    /// text of `expr=` and the spaces around it when the field is written
    /// `{expr=}`; the conversion (`r` or `s`); the format spec, each field
    /// nested in it (`{x:>{w}}`, `{x:.{p}f}`) written `*`, and the nested
    /// fields' tokens, in order.
    Field { toks: Vec<Token>, eq: Option<String>, conv: Option<char>, spec: Option<String>, spec_fields: Vec<Vec<Token>>, span: Span },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub tok: Tok,
    pub span: Span,
}

pub fn lex(src: &str, source: SourceId) -> Result<Vec<Token>, Vec<Diagnostic>> {
    one_line_steps(lex_range(src, 0, src.len(), source)?)
}

/// Whether `toks[k..]` starts a one-line step, `step <id> ["<text>"] :`, and
/// the index of its `:`. Only at the start of a statement: after a line
/// break, after a `{`, or first in the file.
fn one_line_step_at(toks: &[Token], k: usize, prev: Option<&Tok>) -> Option<usize> {
    if !matches!(prev, None | Some(Tok::Newline) | Some(Tok::LBrace)) {
        return None;
    }
    let tok = |j: usize| toks.get(j).map(|t| &t.tok);
    if !matches!(tok(k), Some(Tok::Ident(s)) if s == "step") || !matches!(tok(k + 1), Some(Tok::Ident(_))) {
        return None;
    }
    let colon = if matches!(tok(k + 2), Some(Tok::Str(_))) { k + 3 } else { k + 2 };
    matches!(tok(colon), Some(Tok::Colon)).then_some(colon)
}

/// The one-line step, `step attn_residual: X1 = X + Y`, is the braced step
/// `step attn_residual { X1 = X + Y }` of one statement, and is written as
/// that here, before anything reads the tokens: the `:` becomes the `{`, and
/// a `}` goes after the statement. Every pass, engine and lens therefore sees
/// the braced step and nothing else.
///
/// The statement starts on the line of the `:` and ends where a statement
/// ends: at the first line break outside brackets, unless the next line
/// starts with `else` (the `if` goes on), or at a `}` that closes the block
/// around the step. A statement that opens a block (`for`, `if`, `while`)
/// therefore runs to the line where its block closes. `;` is a line break
/// in Viv, so `step a: x = 1; y = 2` holds `x = 1` only.
fn one_line_steps(toks: Vec<Token>) -> Result<Vec<Token>, Vec<Diagnostic>> {
    if !toks.iter().any(|t| matches!(&t.tok, Tok::Ident(s) if s == "step")) {
        return Ok(toks);
    }
    let mut out: Vec<Token> = Vec::with_capacity(toks.len() + 8);
    // each open one-line step: the bracket depth it began at and its `:`
    let mut open: Vec<(i32, Span)> = Vec::new();
    let mut depth: i32 = 0;
    let mut k = 0;
    while k < toks.len() {
        let t = &toks[k];
        // close every one-line step this token ends, innermost first
        while let Some(&(d, colon)) = open.last() {
            let ends = depth == d
                && match t.tok {
                    Tok::Eof | Tok::RBrace => true,
                    Tok::Newline => {
                        let mut j = k;
                        while matches!(toks.get(j).map(|t| &t.tok), Some(Tok::Newline)) {
                            j += 1;
                        }
                        !matches!(toks.get(j).map(|t| &t.tok), Some(Tok::Else))
                    }
                    _ => false,
                };
            if !ends {
                break;
            }
            let last = out.last().expect("the step's `{`");
            if last.tok == Tok::LBrace && last.span == colon {
                return Err(vec![Diagnostic::error("a one-line step holds one statement, written after its `:` on the same line")
                    .with_primary(colon, "nothing follows on this line")
                    .with_note("`step name: x = f(y)`; a step of several statements, or one whose statement starts on the next line, is written with braces: `step name { ... }`")]);
            }
            let end = last.span.end;
            out.push(Token { tok: Tok::RBrace, span: Span { start: end, end, ..last.span } });
            open.pop();
        }
        if let Some(colon) = one_line_step_at(&toks, k, out.last().map(|t| &t.tok)) {
            out.extend(toks[k..colon].iter().cloned());
            out.push(Token { tok: Tok::LBrace, span: toks[colon].span });
            open.push((depth, toks[colon].span));
            k = colon + 1;
            continue;
        }
        match t.tok {
            Tok::LParen | Tok::LBracket | Tok::LBrace => depth += 1,
            Tok::RParen | Tok::RBracket | Tok::RBrace => depth -= 1,
            _ => {}
        }
        out.push(t.clone());
        k += 1;
    }
    Ok(out)
}

/// The tokens of `src[start..end]`, with spans in `src`: the whole program,
/// or the expression of an f-string's field.
fn lex_range(src: &str, start: usize, end: usize, source: SourceId) -> Result<Vec<Token>, Vec<Diagnostic>> {
    let src = &src[..end];
    let bytes = src.as_bytes();
    let n = end;
    let mut i = start;
    let mut toks = Vec::new();
    let mut errs = Vec::new();
    // Newlines inside `(...)` are continuation whitespace (implicit line join).
    let mut paren_depth: i32 = 0;

    let at = |start: usize, end: usize| Span::new(source, start as u32, end as u32);

    while i < n {
        // `i` is always on a character boundary: every branch below steps
        // over ASCII bytes or, for an identifier or a stray character, whole
        // characters, so a name like `Δx` lexes as one identifier.
        let c = src[i..].chars().next().unwrap();
        match c {
            '\n' => {
                if paren_depth == 0 {
                    toks.push(Token { tok: Tok::Newline, span: at(i, i + 1) });
                }
                i += 1;
            }
            ';' => {
                toks.push(Token { tok: Tok::Newline, span: at(i, i + 1) });
                i += 1;
            }
            c if c.is_whitespace() => i += 1,
            '#' => {
                // A FENCE: a line that is exactly `###` opens a block that runs
                // to the next such line (or to the end of the file). Everything
                // inside is text, not program -- which is how a post carries its
                // title and its prose without prefixing every line with `#`.
                let line_start = src[..i].rfind('\n').map_or(0, |p| p + 1);
                let line_end = src[i..].find('\n').map_or(n, |p| i + p);
                let is_fence = src[line_start..i].trim().is_empty() && src[i..line_end].trim_end() == "###";
                if is_fence {
                    i = line_end;
                    while i < n {
                        let start = i + 1;
                        let end = src[start..].find('\n').map_or(n, |p| start + p);
                        if start >= n { i = n; break; }
                        if src[start..end].trim_end() == "###" { i = end; break; }
                        i = end;
                    }
                    if i < n { i += 1; }
                } else {
                    while i < n && bytes[i] != b'\n' {
                        i += 1;
                    }
                }
            }
            '(' => { toks.push(Token { tok: Tok::LParen, span: at(i, i + 1) }); paren_depth += 1; i += 1; }
            ')' => { toks.push(Token { tok: Tok::RParen, span: at(i, i + 1) }); paren_depth -= 1; i += 1; }
            '{' => { toks.push(Token { tok: Tok::LBrace, span: at(i, i + 1) }); i += 1; }
            '}' => { toks.push(Token { tok: Tok::RBrace, span: at(i, i + 1) }); i += 1; }
            '[' => { toks.push(Token { tok: Tok::LBracket, span: at(i, i + 1) }); paren_depth += 1; i += 1; }
            ']' => { toks.push(Token { tok: Tok::RBracket, span: at(i, i + 1) }); paren_depth -= 1; i += 1; }
            ',' => { toks.push(Token { tok: Tok::Comma, span: at(i, i + 1) }); i += 1; }
            '+' if i + 1 < n && bytes[i + 1] == b'=' => { toks.push(Token { tok: Tok::PlusEq, span: at(i, i + 2) }); i += 2; }
            '+' => { toks.push(Token { tok: Tok::Plus, span: at(i, i + 1) }); i += 1; }
            '-' if i + 1 < n && bytes[i + 1] == b'=' => { toks.push(Token { tok: Tok::MinusEq, span: at(i, i + 2) }); i += 2; }
            '-' if i + 1 < n && bytes[i + 1] == b'>' => { toks.push(Token { tok: Tok::Arrow, span: at(i, i + 2) }); i += 2; }
            '-' => { toks.push(Token { tok: Tok::Minus, span: at(i, i + 1) }); i += 1; }
            '*' if i + 1 < n && bytes[i + 1] == b'=' => { toks.push(Token { tok: Tok::StarEq, span: at(i, i + 2) }); i += 2; }
            '*' => { toks.push(Token { tok: Tok::Star, span: at(i, i + 1) }); i += 1; }
            '/' if i + 2 < n && bytes[i + 1] == b'/' && bytes[i + 2] == b'=' => { toks.push(Token { tok: Tok::SlashSlashEq, span: at(i, i + 3) }); i += 3; }
            '/' if i + 1 < n && bytes[i + 1] == b'/' => { toks.push(Token { tok: Tok::SlashSlash, span: at(i, i + 2) }); i += 2; }
            '/' if i + 1 < n && bytes[i + 1] == b'=' => { toks.push(Token { tok: Tok::SlashEq, span: at(i, i + 2) }); i += 2; }
            '/' => { toks.push(Token { tok: Tok::Slash, span: at(i, i + 1) }); i += 1; }
            '@' => { toks.push(Token { tok: Tok::At, span: at(i, i + 1) }); i += 1; }
            '%' => { toks.push(Token { tok: Tok::Percent, span: at(i, i + 1) }); i += 1; }
            '^' => { toks.push(Token { tok: Tok::Caret, span: at(i, i + 1) }); i += 1; }
            ':' => { toks.push(Token { tok: Tok::Colon, span: at(i, i + 1) }); i += 1; }
            // A `.` not flanked by digits is field access (floats need a digit on
            // both sides; the digit branch handles `1.5`).
            // `.5`: a number, where a `.` cannot be field access (nothing that
            // has fields comes before it)
            '.' if i + 1 < n
                && (bytes[i + 1] as char).is_ascii_digit()
                && !matches!(toks.last().map(|t| &t.tok), Some(Tok::Ident(_) | Tok::RParen | Tok::RBracket | Tok::Num(..) | Tok::Str(_) | Tok::RBrace)) =>
            {
                let start = i;
                i += 1;
                while i < n && ((bytes[i] as char).is_ascii_digit() || (bytes[i] == b'_' && i + 1 < n && (bytes[i + 1] as char).is_ascii_digit())) {
                    i += 1;
                }
                if i < n && (bytes[i] == b'e' || bytes[i] == b'E') {
                    let mut j = i + 1;
                    if j < n && (bytes[j] == b'+' || bytes[j] == b'-') {
                        j += 1;
                    }
                    if j < n && (bytes[j] as char).is_ascii_digit() {
                        i = j;
                        while i < n && (bytes[i] as char).is_ascii_digit() {
                            i += 1;
                        }
                    }
                }
                let text: String = src[start..i].chars().filter(|&c| c != '_').collect();
                match format!("0{text}").parse::<f64>() {
                    Ok(v) => toks.push(Token { tok: Tok::Num(v, None), span: at(start, i) }),
                    Err(_) => errs.push(Diagnostic::error("invalid number literal").with_primary(at(start, i), "here")),
                }
            }
            '.' => { toks.push(Token { tok: Tok::Dot, span: at(i, i + 1) }); i += 1; }
            '"' | '\'' => match string_literal(src, i, source) {
                Ok((value, end)) => {
                    toks.push(Token { tok: Tok::Str(value), span: at(i, end) });
                    i = end;
                }
                Err(e) => {
                    errs.push(e);
                    i = n;
                }
            },
            '=' => {
                if i + 1 < n && bytes[i + 1] == b'=' {
                    toks.push(Token { tok: Tok::EqEq, span: at(i, i + 2) });
                    i += 2;
                } else if i + 1 < n && bytes[i + 1] == b'>' {
                    toks.push(Token { tok: Tok::FatArrow, span: at(i, i + 2) });
                    i += 2;
                } else {
                    toks.push(Token { tok: Tok::Eq, span: at(i, i + 1) });
                    i += 1;
                }
            }
            '<' => {
                if i + 1 < n && bytes[i + 1] == b'=' {
                    toks.push(Token { tok: Tok::Le, span: at(i, i + 2) });
                    i += 2;
                } else {
                    toks.push(Token { tok: Tok::Lt, span: at(i, i + 1) });
                    i += 1;
                }
            }
            '>' => {
                if i + 1 < n && bytes[i + 1] == b'=' {
                    toks.push(Token { tok: Tok::Ge, span: at(i, i + 2) });
                    i += 2;
                } else {
                    toks.push(Token { tok: Tok::Gt, span: at(i, i + 1) });
                    i += 1;
                }
            }
            '!' => {
                if i + 1 < n && bytes[i + 1] == b'=' {
                    toks.push(Token { tok: Tok::Ne, span: at(i, i + 2) });
                    i += 2;
                } else {
                    toks.push(Token { tok: Tok::Not, span: at(i, i + 1) });
                    i += 1;
                }
            }
            '&' if i + 1 < n && bytes[i + 1] == b'&' => {
                toks.push(Token { tok: Tok::And, span: at(i, i + 2) });
                i += 2;
            }
            '|' if i + 1 < n && bytes[i + 1] == b'|' => {
                toks.push(Token { tok: Tok::Or, span: at(i, i + 2) });
                i += 2;
            }
            '|' => {
                toks.push(Token { tok: Tok::Pipe, span: at(i, i + 1) });
                i += 1;
            }
            '?' => {
                toks.push(Token { tok: Tok::Question, span: at(i, i + 1) });
                i += 1;
            }
            // `0xff`, `0o17`, `0b101`: an integer literal in base 16, 8 or 2,
            // as Python writes them (`_` between digits allowed)
            '0' if i + 2 < n && matches!(bytes[i + 1], b'x' | b'X' | b'o' | b'O' | b'b' | b'B') && (bytes[i + 2] as char).is_ascii_hexdigit() => {
                let start = i;
                let radix = match bytes[i + 1] {
                    b'x' | b'X' => 16,
                    b'o' | b'O' => 8,
                    _ => 2,
                };
                i += 2;
                while i < n && ((bytes[i] as char).is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                let digits: String = src[start + 2..i].chars().filter(|&c| c != '_').collect();
                let span = at(start, i);
                match u64::from_str_radix(&digits, radix) {
                    Ok(v) => toks.push(Token { tok: Tok::Num(v as f64, i64::try_from(v).ok()), span }),
                    Err(_) => errs.push(Diagnostic::error(format!("`{}` is not a number in base {radix}", &src[start..i])).with_primary(span, "here")),
                }
            }
            c if c.is_ascii_digit() => {
                let start = i;
                // `1_000_000`: Python's separators, between digits
                let digit_or_sep = |k: usize| (bytes[k] as char).is_ascii_digit() || (bytes[k] == b'_' && k + 1 < n && (bytes[k + 1] as char).is_ascii_digit());
                while i < n && digit_or_sep(i) {
                    i += 1;
                }
                // A `.` is part of the float only with a digit (or an exponent,
                // `2.e5`) after it; otherwise it is field access (`1.5` is a
                // float, `x.y` and `1.foo` are not).
                if i + 1 < n && bytes[i] == b'.' && (bytes[i + 1] as char).is_ascii_digit() {
                    i += 1;
                    while i < n && digit_or_sep(i) {
                        i += 1;
                    }
                } else if i + 2 < n && bytes[i] == b'.' && matches!(bytes[i + 1], b'e' | b'E') && ((bytes[i + 2] as char).is_ascii_digit() || (matches!(bytes[i + 2], b'+' | b'-') && i + 3 < n && (bytes[i + 3] as char).is_ascii_digit())) {
                    i += 1;
                }
                // Optional exponent, e.g. 1e-10.
                if i < n && (bytes[i] == b'e' || bytes[i] == b'E') {
                    let mut j = i + 1;
                    if j < n && (bytes[j] == b'+' || bytes[j] == b'-') {
                        j += 1;
                    }
                    if j < n && (bytes[j] as char).is_ascii_digit() {
                        i = j;
                        while i < n && (bytes[i] as char).is_ascii_digit() {
                            i += 1;
                        }
                    }
                }
                let text = &src[start..i];
                // An `i` suffix makes an imaginary literal (`2.5i`). Only on a
                // numeric literal - a bare identifier `i` still lexes as a name.
                let imag = i < n
                    && bytes[i] == b'i'
                    && !(i + 1 < n && ((bytes[i + 1] as char).is_alphanumeric() || bytes[i + 1] == b'_'));
                if imag {
                    i += 1;
                }
                let span = at(start, i);
                // `2x`: a name right after a number. Read as two statements it
                // gives silently wrong numbers (`y = 2x` sets y to 2).
                if src[i..].chars().next().is_some_and(|ch| ch.is_alphabetic() || ch == '_') {
                    let end = i + src[i..].find(|ch: char| !(ch.is_alphanumeric() || ch == '_')).unwrap_or(n - i);
                    let name = &src[i..end];
                    errs.push(
                        Diagnostic::error(format!("a number cannot be followed directly by a name: `{}{}`", text, name))
                            .with_primary(at(start, end), "here")
                            .with_note(crate::hints::juxtapose(text, name)),
                    );
                    i = end;
                    continue;
                }
                let clean: String = text.chars().filter(|&c| c != '_').collect();
                let text = clean.as_str();
                let whole = if text.contains(['.', 'e', 'E']) { None } else { text.parse::<i64>().ok() };
                match text.trim_end_matches('.').parse::<f64>().or_else(|_| text.replace(".e", "e").replace(".E", "E").parse::<f64>()) {
                    Ok(v) => toks.push(Token {
                        tok: if imag { Tok::Imag(v) } else { Tok::Num(v, whole) },
                        span,
                    }),
                    Err(_) => errs.push(
                        Diagnostic::error("invalid number literal").with_primary(span, "here"),
                    ),
                }
            }
            // `f"..."`: an f-string, not the name `f` and a string
            'f' if i + 1 < n && bytes[i + 1] == b'"' && !(i > 0 && (src[..i].chars().next_back().is_some_and(|p| p.is_alphanumeric() || p == '_'))) => {
                if src[i + 1..].starts_with("\"\"\"") {
                    errs.push(Diagnostic::error("a triple-quoted f-string is not supported").with_primary(at(i, i + 4), "here").with_note("join a triple-quoted string and an f-string: \"\"\"...\"\"\" f\"{x}\""));
                    break;
                }
                match fstring(src, i + 1, source) {
                    Ok((parts, end)) => {
                        toks.push(Token { tok: Tok::FStr(parts), span: at(i, end) });
                        i = end;
                    }
                    Err(e) => {
                        errs.extend(e);
                        i = n;
                    }
                }
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while let Some(ch) = src[i..].chars().next().filter(|ch| ch.is_alphanumeric() || *ch == '_') {
                    i += ch.len_utf8();
                }
                let text = &src[start..i];
                let span = at(start, i);
                let tok = match text {
                    "def" => Tok::Def,
                    "return" => Tok::Return,
                    "if" => Tok::If,
                    "else" => Tok::Else,
                    "while" => Tok::While,
                    "for" => Tok::For,
                    "to" => Tok::To,
                    "break" => Tok::Break,
                    "continue" => Tok::Continue,
                    "narrate" => Tok::Narrate,
                    "show" => Tok::Show,
                    "test" => Tok::Test,
                    "rung" => Tok::Rung,
                    "parallel" => Tok::Parallel,
                    "fn" => Tok::Fn,
                    "record" => Tok::Record,
                    "and" => Tok::And,
                    "or" => Tok::Or,
                    "not" => Tok::Not,
                    "true" => Tok::True,
                    "false" => Tok::False,
                    _ => Tok::Ident(text.to_string()),
                };
                toks.push(Token { tok, span });
            }
            other => {
                // `%` is an operator now, so no character here needs a hint
                let w = other.len_utf8();
                errs.push(Diagnostic::error(format!("unexpected character {:?}", other)).with_primary(at(i, i + w), "here"));
                i += w;
            }
        }
    }

    toks.push(Token { tok: Tok::Eof, span: at(n, n) });

    if errs.is_empty() {
        Ok(toks)
    } else {
        Err(errs)
    }
}


/// One escape after a backslash at `bytes[i - 1]`: the character, and where
/// the text goes on. `\n` `\t` `\r` `\0` `\\` `\"` `\'` and `\u{XXXX}` (one to
/// six hex digits, a Unicode scalar value).
fn escape(src: &str, i: usize, source: SourceId) -> Result<(char, usize), Diagnostic> {
    let bytes = src.as_bytes();
    let at = |a: usize, b: usize| Span::new(source, a as u32, b as u32);
    let Some(&b) = bytes.get(i) else {
        return Err(Diagnostic::error("unterminated string literal").with_primary(at(i - 1, i), "a backslash at the end"));
    };
    let c = match b {
        b'n' => '\n',
        b't' => '\t',
        b'r' => '\r',
        b'0' => '\0',
        b'\\' => '\\',
        b'"' => '"',
        b'\'' => '\'',
        // `\xHH`: the character with that code, below 256, as Python reads it
        b'x' => {
            let hex = src.get(i + 1..i + 3).unwrap_or("");
            return match (hex.len() == 2).then(|| u32::from_str_radix(hex, 16).ok()).flatten().and_then(char::from_u32) {
                Some(c) => Ok((c, i + 3)),
                None => Err(Diagnostic::error("a `\\x` escape is written `\\x` and two hex digits").with_primary(at(i - 1, (i + 3).min(src.len())), "here")),
            };
        }
        b'u' => {
            let close = if bytes.get(i + 1) == Some(&b'{') { src[i + 2..].find('}').map(|k| i + 2 + k) } else { None };
            let Some(close) = close else {
                return Err(Diagnostic::error("a `\\u` escape is written `\\u{...}` with one to six hex digits").with_primary(at(i - 1, i + 1), "here"));
            };
            let hex = &src[i + 2..close];
            let v = if (1..=6).contains(&hex.len()) { u32::from_str_radix(hex, 16).ok() } else { None };
            return match v.and_then(char::from_u32) {
                Some(c) => Ok((c, close + 1)),
                None => Err(Diagnostic::error(format!("`\\u{{{hex}}}` is not a Unicode character"))
                    .with_primary(at(i - 1, close + 1), "not a character")
                    .with_note("a character is a hex number up to 10FFFF, and not a surrogate (D800 to DFFF)")),
            };
        }
        _ => {
            let c = src[i..].chars().next().expect("a character after the backslash");
            return Err(Diagnostic::error(format!("unknown escape `\\{c}` in a string"))
                .with_primary(at(i - 1, i + c.len_utf8()), "not an escape")
                .with_note("the escapes are \\n \\t \\r \\0 \\\\ \\\" \\' \\xHH and \\u{...}; write \\\\ for a backslash"));
        }
    };
    Ok((c, i + 1))
}

/// A string literal whose opening quote is at `start`: its text, and the
/// position after the closing quote.
fn string_literal(src: &str, start: usize, source: SourceId) -> Result<(String, usize), Diagnostic> {
    let bytes = src.as_bytes();
    let mut out = String::new();
    // `"""..."""`: a triple-quoted string, Python's, which may hold newlines
    // and single quotes; escapes as in any string
    let q = bytes[start];
    let triple = if q == b'"' { "\"\"\"" } else { "'''" };
    if src[start..].starts_with(triple) {
        let mut i = start + 3;
        loop {
            if src[i..].starts_with(triple) {
                return Ok((out, i + 3));
            }
            let Some(&b) = bytes.get(i) else {
                return Err(Diagnostic::error("unterminated triple-quoted string").with_primary(Span::new(source, start as u32, src.len() as u32), "starts here"));
            };
            if b == b'\\' {
                let (c, next) = escape(src, i + 1, source)?;
                out.push(c);
                i = next;
            } else {
                let c = src[i..].chars().next().expect("a character");
                out.push(c);
                i += c.len_utf8();
            }
        }
    }
    let mut i = start + 1;
    loop {
        let Some(&b) = bytes.get(i) else {
            return Err(Diagnostic::error("unterminated string literal").with_primary(Span::new(source, start as u32, src.len() as u32), "starts here"));
        };
        match b {
            b if b == q => return Ok((out, i + 1)),
            b'\\' => {
                let (c, next) = escape(src, i + 1, source)?;
                out.push(c);
                i = next;
            }
            _ => {
                let c = src[i..].chars().next().expect("a character");
                out.push(c);
                i += c.len_utf8();
            }
        }
    }
}

/// Where a string literal that opens at `q` ends (after its closing quote),
/// for finding the end of an f-string field that holds one. An f-string
/// (`f` just before the quote) may hold fields with literals of their own.
fn skip_literal(src: &str, q: usize) -> Option<usize> {
    let bytes = src.as_bytes();
    let is_f = q > 0 && bytes[q - 1] == b'f' && !(q > 1 && (bytes[q - 2].is_ascii_alphanumeric() || bytes[q - 2] == b'_'));
    let mut i = q + 1;
    let mut depth = 0usize;
    loop {
        match *bytes.get(i)? {
            b'\\' => i += 2,
            b'"' if depth == 0 => return Some(i + 1),
            b'"' => i = skip_literal(src, i)?,
            b'{' if is_f => {
                if depth == 0 && bytes.get(i + 1) == Some(&b'{') {
                    i += 2;
                } else {
                    depth += 1;
                    i += 1;
                }
            }
            b'}' if is_f && depth > 0 => {
                depth -= 1;
                i += 1;
            }
            _ => i += 1,
        }
    }
}

/// An f-string whose opening quote is at `q`: its parts and the position after
/// its closing quote. A field is `{expr}`, `{expr=}`, `{expr!r}`, `{expr!s}`,
/// `{expr:spec}`, or several of those in that order; `{{` and `}}` are braces.
/// A field may hold string literals, and other f-strings (PEP 701's rule), so
/// `f"{join(", ", names)}"` is one f-string.
fn fstring(src: &str, q: usize, source: SourceId) -> Result<(Vec<FPart>, usize), Vec<Diagnostic>> {
    let bytes = src.as_bytes();
    let at = |a: usize, b: usize| Span::new(source, a as u32, b as u32);
    let unterminated = || vec![Diagnostic::error("unterminated f-string").with_primary(at(q - 1, src.len()), "starts here")];
    let mut parts = Vec::new();
    let mut text = String::new();
    let mut i = q + 1;
    loop {
        let Some(&b) = bytes.get(i) else { return Err(unterminated()) };
        match b {
            b'"' => {
                if !text.is_empty() {
                    parts.push(FPart::Text(text));
                }
                return Ok((parts, i + 1));
            }
            b'\\' => {
                let (c, next) = escape(src, i + 1, source).map_err(|d| vec![d])?;
                text.push(c);
                i = next;
            }
            b'{' if bytes.get(i + 1) == Some(&b'{') => {
                text.push('{');
                i += 2;
            }
            b'}' if bytes.get(i + 1) == Some(&b'}') => {
                text.push('}');
                i += 2;
            }
            b'}' => {
                return Err(vec![Diagnostic::error("a single `}` is not allowed in an f-string")
                    .with_primary(at(i, i + 1), "write `}}` for a brace")]);
            }
            b'{' => {
                if !text.is_empty() {
                    parts.push(FPart::Text(std::mem::take(&mut text)));
                }
                let (part, next) = field(src, i, source)?;
                parts.push(part);
                i = next;
            }
            _ => {
                let c = src[i..].chars().next().expect("a character");
                text.push(c);
                i += c.len_utf8();
            }
        }
    }
}

/// One field of an f-string, its `{` at `open`: the part, and the position
/// after its `}`.
fn field(src: &str, open: usize, source: SourceId) -> Result<(FPart, usize), Vec<Diagnostic>> {
    let bytes = src.as_bytes();
    let at = |a: usize, b: usize| Span::new(source, a as u32, b as u32);
    let err = |a: usize, b: usize, m: &str, label: &str| vec![Diagnostic::error(m.to_string()).with_primary(at(a, b), label.to_string())];
    let start = open + 1;
    // the end of the expression: a `}`, `!`, `:` or a lone `=` outside brackets
    let mut i = start;
    let mut depth = 0i32;
    let expr_end;
    let mut eq_at = None;
    loop {
        let Some(&b) = bytes.get(i) else { return Err(err(open, src.len(), "unterminated f-string field", "opens here")) };
        match b {
            b'"' => i = skip_literal(src, i).ok_or_else(|| err(open, src.len(), "unterminated string in an f-string field", "in this field"))?,
            b'(' | b'[' | b'{' => {
                depth += 1;
                i += 1;
            }
            b')' | b']' => {
                depth -= 1;
                i += 1;
            }
            b'}' if depth > 0 => {
                depth -= 1;
                i += 1;
            }
            b'#' => return Err(err(i, i + 1, "an f-string field cannot hold a comment", "here")),
            b'}' | b':' if depth == 0 => {
                expr_end = i;
                break;
            }
            b'!' if depth == 0 && bytes.get(i + 1) != Some(&b'=') => {
                expr_end = i;
                break;
            }
            b'=' if depth == 0
                && bytes.get(i + 1) != Some(&b'=')
                && !matches!(bytes.get(i.wrapping_sub(1)), Some(b'=' | b'!' | b'<' | b'>')) =>
            {
                // `{x=}`: the rest of the field is spaces, then `}`, `!` or `:`
                let mut k = i + 1;
                while matches!(bytes.get(k), Some(b' ')) {
                    k += 1;
                }
                if !matches!(bytes.get(k), Some(b'}' | b'!' | b':')) {
                    return Err(err(i, i + 1, "`=` in an f-string field is `{expr=}`", "an assignment cannot go here"));
                }
                eq_at = Some(i);
                expr_end = k;
                break;
            }
            _ => i += 1,
        }
    }
    let code_end = eq_at.unwrap_or(expr_end);
    if src[start..code_end].trim().is_empty() {
        return Err(err(open, expr_end + 1, "an f-string field needs an expression", "empty field"));
    }
    let eq = eq_at.map(|_| src[start..expr_end].to_string());
    let mut toks = lex_range(src, start, code_end, source)?;
    // a field's expression is one line: no statement ends inside it
    toks.retain(|t| t.tok != Tok::Newline);
    toks.push(Token { tok: Tok::Eof, span: at(code_end, code_end) });
    let mut i = expr_end;
    let mut conv = None;
    if bytes.get(i) == Some(&b'!') {
        match bytes.get(i + 1) {
            Some(b'r') => conv = Some('r'),
            Some(b's') => conv = Some('s'),
            Some(b'a') => return Err(err(i, i + 2, "`!a` (ascii) is not supported in an f-string", "use `!r`")),
            _ => return Err(err(i, i + 2, "an f-string conversion is `!r` or `!s`", "here")),
        }
        i += 2;
    }
    let mut spec = None;
    let mut spec_fields = Vec::new();
    if bytes.get(i) == Some(&b':') {
        let mut k = i + 1;
        let mut text = String::new();
        loop {
            match bytes.get(k) {
                None => return Err(err(open, src.len(), "unterminated f-string field", "opens here")),
                Some(b'}') => break,
                // a nested field, Python's `{x:>{w}}`: one expression, the
                // spec's width or precision
                Some(b'{') => {
                    let e0 = k + 1;
                    let mut e1 = e0;
                    while !matches!(bytes.get(e1), Some(b'}' | b'{') | None) {
                        e1 += 1;
                    }
                    if bytes.get(e1) != Some(&b'}') || src[e0..e1].trim().is_empty() {
                        return Err(err(k, e1.min(src.len()), "a field in a format spec is `{name}`, one expression in braces", "here"));
                    }
                    let mut ft = lex_range(src, e0, e1, source)?;
                    ft.retain(|t| t.tok != Tok::Newline);
                    ft.push(Token { tok: Tok::Eof, span: at(e1, e1) });
                    spec_fields.push(ft);
                    text.push('*');
                    k = e1 + 1;
                }
                Some(_) => {
                    let c = src[k..].chars().next().expect("a character");
                    text.push(c);
                    k += c.len_utf8();
                }
            }
        }
        spec = Some(text);
        i = k;
    }
    if bytes.get(i) != Some(&b'}') {
        return Err(err(i, i + 1, "expected `}` to end the f-string field", "here"));
    }
    Ok((FPart::Field { toks, eq, conv, spec, spec_fields, span: at(open, i + 1) }, i + 1))
}
