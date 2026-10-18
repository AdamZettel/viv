// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Mathtext: the `$...$` parts of a figure label set as math, in SVG, with no
//! TeX installation (as matplotlib's mathtext does).
//!
//! A label such as `$E/A$ (MeV)` or `$\frac{dE}{dt}$` is parsed as a subset
//! of TeX and laid out with the metrics of the KaTeX fonts (the site sets its
//! math in KaTeX; mathtext_metrics.rs is generated from KaTeX's own table).
//! The result is the content of one SVG `<text>`: a `<tspan>` per run, moved
//! with `dx`/`dy` and sized for sub- and superscripts, so the browser places
//! the runs one after another and `text-anchor` still centers the whole label.
//! Fraction bars, radicals and overlines are drawn beside the text as shapes.
//! Text outside the dollars stays in the figure's font; `\$` is a dollar sign.
//! A label with an odd number of dollars has no math (as in matplotlib).
//!
//! The supported subset: Greek letters (\alpha..\omega with the var forms,
//! \Gamma..\Omega), `^` and `_` (nested), `'` for a prime, \frac \tfrac
//! \dfrac (full size), \sqrt, \overline, \hat \bar \vec \dot \ddot \tilde, \mathrm
//! \mathbf \mathit \text \operatorname, the operator names (\sin \cos \tan
//! \exp \log \ln \max \min \lim \det ... set upright), \sum \prod \int \oint,
//! \left and \right, the spacing commands (\, \: \; \! \quad \qquad), and the
//! symbols in SYMBOLS below. An unknown command is an error with its column.
//!
//! www/src/lib/mathtext.js is the same code for the site's own charts; the
//! test www/tests/mathtext.test.mjs checks both give the same output.

use crate::mathtext_metrics::FONTS;
use std::fmt::Write;

/// The KaTeX font a glyph is set in (the order of mathtext_metrics::FONTS).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Font {
    Main,
    Bold,
    Italic,
    Math,
    MathBold,
    Size1,
}

/// TeX's atom classes, for the spacing between atoms.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Cls {
    Ord,
    Op,
    Bin,
    Rel,
    Open,
    Close,
    Punct,
    Inner,
}

/// The font stacks: KaTeX's faces first (loaded on the site's pages), then
/// close serif faces for an SVG opened on its own.
const MAIN_STACK: &str = "KaTeX_Main, 'Latin Modern Roman', 'Times New Roman', serif";
const MATH_STACK: &str = "KaTeX_Math, 'Latin Modern Math', 'Times New Roman', serif";
const SIZE1_STACK: &str = "KaTeX_Size1, KaTeX_Main, 'Times New Roman', serif";

/// Math is set this much larger than the label's text, as KaTeX does beside
/// sans-serif text (its fonts have a smaller x-height).
pub const MATH_SCALE: f64 = 1.15;
/// Font size of text, script and scriptscript, relative to text.
const SCALES: [f64; 3] = [1.0, 0.7, 0.5];
// TeX's font parameters for text style, in em (KaTeX src/fontMetrics.ts).
const X_HEIGHT: f64 = 0.431;
const NUM1: f64 = 0.677;
const NUM2: f64 = 0.394;
const DENOM1: f64 = 0.686;
const DENOM2: f64 = 0.345;
const SUP1: f64 = 0.413;
const SUB1: f64 = 0.15;
const SUB2: f64 = 0.247;
const SUP_DROP: f64 = 0.386;
const SUB_DROP: f64 = 0.05;
const AXIS: f64 = 0.25;
const RULE: f64 = 0.04;
const SCRIPT_SPACE: f64 = 0.05;
const NULL_DELIM: f64 = 0.12;
/// One math unit: an eighteenth of an em.
const MU: f64 = 1.0 / 18.0;

/// A parse error: the column (1-based, in characters of the whole label) and what is wrong.
#[derive(Debug, Clone, PartialEq)]
pub struct MathError {
    pub col: usize,
    pub msg: String,
}

impl std::fmt::Display for MathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at column {}", self.msg, self.col)
    }
}

#[derive(Clone, Debug)]
enum Node {
    Glyph { ch: char, font: Font, cls: Cls },
    Word { text: String, font: Font, cls: Cls },
    Group(Vec<Node>),
    Scripts { base: Box<Node>, sup: Option<Vec<Node>>, sub: Option<Vec<Node>> },
    /// numerator, denominator, and whether in display style (\dfrac)
    Frac(Vec<Node>, Vec<Node>, bool),
    Sqrt(Vec<Node>),
    Overline(Vec<Node>),
    Accent { ch: char, body: Vec<Node> },
    Space(f64),
    Delimited { left: Option<char>, body: Vec<Node>, right: Option<char> },
}

impl Node {
    fn cls(&self) -> Option<Cls> {
        match self {
            Node::Glyph { cls, .. } | Node::Word { cls, .. } => Some(*cls),
            Node::Scripts { base, .. } => base.cls(),
            Node::Frac(..) | Node::Delimited { .. } => Some(Cls::Inner),
            Node::Space(_) => None,
            _ => Some(Cls::Ord),
        }
    }
}

/// A command that stands for one glyph: its character, class and font.
fn symbol(name: &str) -> Option<(char, Cls, Font)> {
    use Cls::*;
    use Font::*;
    let greek_lower = [
        ("alpha", 'α'), ("beta", 'β'), ("gamma", 'γ'), ("delta", 'δ'), ("epsilon", 'ϵ'), ("varepsilon", 'ε'),
        ("zeta", 'ζ'), ("eta", 'η'), ("theta", 'θ'), ("vartheta", 'ϑ'), ("iota", 'ι'), ("kappa", 'κ'),
        ("lambda", 'λ'), ("mu", 'μ'), ("nu", 'ν'), ("xi", 'ξ'), ("pi", 'π'), ("varpi", 'ϖ'), ("rho", 'ρ'),
        ("varrho", 'ϱ'), ("sigma", 'σ'), ("varsigma", 'ς'), ("tau", 'τ'), ("upsilon", 'υ'), ("phi", 'ϕ'),
        ("varphi", 'φ'), ("chi", 'χ'), ("psi", 'ψ'), ("omega", 'ω'),
    ];
    if let Some((_, c)) = greek_lower.iter().find(|(n, _)| *n == name) {
        return Some((*c, Ord, Math));
    }
    let table: &[(&str, char, Cls)] = &[
        ("Gamma", 'Γ', Ord), ("Delta", 'Δ', Ord), ("Theta", 'Θ', Ord), ("Lambda", 'Λ', Ord), ("Xi", 'Ξ', Ord),
        ("Pi", 'Π', Ord), ("Sigma", 'Σ', Ord), ("Upsilon", 'Υ', Ord), ("Phi", 'Φ', Ord), ("Psi", 'Ψ', Ord),
        ("Omega", 'Ω', Ord),
        ("partial", '∂', Ord), ("nabla", '∇', Ord), ("infty", '∞', Ord), ("hbar", 'ℏ', Ord), ("ell", 'ℓ', Ord),
        ("prime", '′', Ord), ("top", '⊤', Ord), ("bot", '⊥', Ord), ("emptyset", '∅', Ord), ("forall", '∀', Ord),
        ("exists", '∃', Ord), ("neg", '¬', Ord), ("ldots", '…', Inner), ("cdots", '⋯', Inner), ("dots", '…', Inner),
        ("degree", '°', Ord), ("|", '∥', Ord), ("Vert", '∥', Ord), ("vert", '∣', Ord), ("star", '⋆', Bin),
        ("pm", '±', Bin), ("mp", '∓', Bin), ("times", '×', Bin), ("cdot", '⋅', Bin), ("div", '÷', Bin),
        ("ast", '∗', Bin), ("circ", '∘', Bin), ("bullet", '∙', Bin), ("oplus", '⊕', Bin), ("otimes", '⊗', Bin),
        ("cup", '∪', Bin), ("cap", '∩', Bin), ("wedge", '∧', Bin), ("vee", '∨', Bin), ("setminus", '∖', Bin),
        ("le", '≤', Rel), ("leq", '≤', Rel), ("ge", '≥', Rel), ("geq", '≥', Rel), ("ne", '≠', Rel), ("neq", '≠', Rel),
        ("approx", '≈', Rel), ("sim", '∼', Rel), ("simeq", '≃', Rel), ("propto", '∝', Rel), ("equiv", '≡', Rel),
        ("ll", '≪', Rel), ("gg", '≫', Rel), ("to", '→', Rel), ("rightarrow", '→', Rel), ("leftarrow", '←', Rel),
        ("gets", '←', Rel), ("Rightarrow", '⇒', Rel), ("Leftarrow", '⇐', Rel), ("leftrightarrow", '↔', Rel),
        ("Leftrightarrow", '⇔', Rel), ("mapsto", '↦', Rel), ("in", '∈', Rel), ("notin", '∉', Rel),
        ("subset", '⊂', Rel), ("subseteq", '⊆', Rel), ("supset", '⊃', Rel), ("supseteq", '⊇', Rel),
        ("perp", '⊥', Rel), ("mid", '∣', Rel), ("parallel", '∥', Rel), ("cong", '≅', Rel),
        ("langle", '⟨', Open), ("rangle", '⟩', Close), ("lvert", '∣', Open), ("rvert", '∣', Close),
        ("lVert", '∥', Open), ("rVert", '∥', Close), ("lfloor", '⌊', Open), ("rfloor", '⌋', Close),
        ("lceil", '⌈', Open), ("rceil", '⌉', Close), ("lbrace", '{', Open), ("rbrace", '}', Close),
        ("{", '{', Open), ("}", '}', Close),
        ("$", '$', Ord), ("%", '%', Ord), ("&", '&', Ord), ("_", '_', Ord), ("#", '#', Ord),
    ];
    if let Some((_, c, k)) = table.iter().find(|(n, _, _)| *n == name) {
        return Some((*c, *k, Main));
    }
    let big = [("sum", '∑'), ("prod", '∏'), ("coprod", '∐'), ("int", '∫'), ("oint", '∮')];
    big.iter().find(|(n, _)| *n == name).map(|(_, c)| (*c, Op, Size1))
}

/// The operator names, set upright with an operator's spacing.
const OPERATORS: &[&str] = &[
    "sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "coth",
    "exp", "log", "ln", "lg", "max", "min", "sup", "inf", "lim", "det", "dim", "ker", "arg", "deg", "gcd", "Pr",
];

/// The accents: the glyph set over the body.
fn accent(name: &str) -> Option<char> {
    match name {
        "hat" => Some('ˆ'),
        "bar" => Some('ˉ'),
        "vec" => Some('→'),
        "dot" => Some('˙'),
        "ddot" => Some('¨'),
        "tilde" => Some('˜'),
        _ => None,
    }
}

/// A plain character in math: its class and its font under the font command in force.
fn char_node(ch: char, style: Font) -> Node {
    use Cls::*;
    let (ch, cls) = match ch {
        '+' => ('+', Bin),
        '-' => ('−', Bin),
        '*' => ('∗', Bin),
        '=' | '<' | '>' | ':' => (ch, Rel),
        ',' | ';' => (ch, Punct),
        '(' | '[' => (ch, Open),
        ')' | ']' | '!' | '?' => (ch, Close),
        _ => (ch, Ord),
    };
    let letter = ch.is_ascii_alphabetic();
    let font = match style {
        Font::Math if letter => Font::Math,
        Font::Math => Font::Main,
        Font::Main => Font::Main,
        Font::Bold => Font::Bold,
        Font::Italic if letter || ch.is_ascii_digit() => Font::Italic,
        Font::Italic => Font::Main,
        _ => Font::Main,
    };
    Node::Glyph { ch, font, cls }
}

struct Parser<'a> {
    s: &'a [char],
    i: usize,
    /// column of s[0] in the whole label, 0-based
    base: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, at: usize, msg: impl Into<String>) -> MathError {
        MathError { col: self.base + at + 1, msg: msg.into() }
    }

    fn skip_space(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_whitespace() {
            self.i += 1;
        }
    }

    /// A command name after its backslash: letters, or one other character.
    fn command(&mut self) -> Result<String, MathError> {
        let start = self.i;
        self.i += 1;
        if self.i >= self.s.len() {
            return Err(self.err(start, "a backslash at the end of the math"));
        }
        if self.s[self.i].is_ascii_alphabetic() {
            let a = self.i;
            while self.i < self.s.len() && self.s[self.i].is_ascii_alphabetic() {
                self.i += 1;
            }
            Ok(self.s[a..self.i].iter().collect())
        }
        else {
            self.i += 1;
            Ok(self.s[self.i - 1].to_string())
        }
    }

    /// A list of atoms up to `}` (when `in_group`), `\right` or the end.
    fn list(&mut self, style: Font, in_group: bool) -> Result<Vec<Node>, MathError> {
        let mut out: Vec<Node> = Vec::new();
        loop {
            self.skip_space();
            if self.i >= self.s.len() {
                return Ok(out);
            }
            let c = self.s[self.i];
            if c == '}' {
                if in_group {
                    return Ok(out);
                }
                return Err(self.err(self.i, "a } with no {"));
            }
            if c == '\\' && self.peek_command("right") {
                return Ok(out);
            }
            if c == '^' || c == '_' || c == '\'' {
                let base = if matches!(out.last(), Some(n) if !matches!(n, Node::Space(_))) { out.pop().unwrap() } else { Node::Group(Vec::new()) };
                let node = self.scripts(base, style)?;
                out.push(node);
                continue;
            }
            let node = self.atom(style)?;
            out.push(node);
        }
    }

    fn peek_command(&self, name: &str) -> bool {
        let n: Vec<char> = name.chars().collect();
        let end = self.i + 1 + n.len();
        self.s.get(self.i) == Some(&'\\')
            && end <= self.s.len()
            && self.s[self.i + 1..end] == n[..]
            && !(end < self.s.len() && self.s[end].is_ascii_alphabetic())
    }

    /// `^`, `_` and `'` after a base: one superscript and one subscript at most.
    fn scripts(&mut self, base: Node, style: Font) -> Result<Node, MathError> {
        let (mut base, mut sup, mut sub) = match base {
            Node::Scripts { base, sup, sub } => (*base, sup, sub),
            b => (b, None, None),
        };
        loop {
            self.skip_space();
            if self.i >= self.s.len() {
                break;
            }
            let at = self.i;
            match self.s[at] {
                '\'' => {
                    self.i += 1;
                    let p = Node::Glyph { ch: '′', font: Font::Main, cls: Cls::Ord };
                    match &mut sup {
                        Some(v) => v.insert(0, p),
                        None => sup = Some(vec![p]),
                    }
                }
                '^' | '_' => {
                    self.i += 1;
                    self.skip_space();
                    if self.i >= self.s.len() || self.s[self.i] == '}' {
                        return Err(self.err(at, format!("{} needs something to raise or lower", self.s[at])));
                    }
                    let arg = self.argument(style)?;
                    let slot = if self.s[at] == '^' { &mut sup } else { &mut sub };
                    match slot {
                        Some(v) if self.s[at] == '^' && !(v.len() == 1 && matches!(v[0], Node::Glyph { ch: '′', .. })) => {
                            return Err(self.err(at, "a second superscript (use braces: x^{ab})"));
                        }
                        Some(_) if self.s[at] == '_' => return Err(self.err(at, "a second subscript (use braces: x_{ab})")),
                        Some(v) => v.extend(arg),
                        None => *slot = Some(arg),
                    }
                }
                _ => break,
            }
        }
        if let Node::Scripts { .. } = base {
            base = Node::Group(vec![base]);
        }
        Ok(Node::Scripts { base: Box::new(base), sup, sub })
    }

    /// One argument: a braced group or a single atom.
    fn argument(&mut self, style: Font) -> Result<Vec<Node>, MathError> {
        self.skip_space();
        if self.i >= self.s.len() {
            return Err(self.err(self.i.saturating_sub(1), "a missing argument"));
        }
        if self.s[self.i] == '{' {
            let open = self.i;
            self.i += 1;
            let v = self.list(style, true)?;
            if self.i >= self.s.len() {
                return Err(self.err(open, "a { with no }"));
            }
            self.i += 1;
            return Ok(v);
        }
        Ok(vec![self.atom(style)?])
    }

    /// The raw text of a braced argument (for \text and \operatorname).
    fn raw_argument(&mut self, what: &str) -> Result<String, MathError> {
        self.skip_space();
        if self.i >= self.s.len() || self.s[self.i] != '{' {
            return Err(self.err(self.i.min(self.s.len().saturating_sub(1)), format!("{what} needs a braced argument")));
        }
        let open = self.i;
        self.i += 1;
        let mut t = String::new();
        while self.i < self.s.len() && self.s[self.i] != '}' {
            let c = self.s[self.i];
            if c == '\\' {
                match self.s.get(self.i + 1) {
                    Some(e) if "${}%&_#".contains(*e) => {
                        t.push(*e);
                        self.i += 2;
                        continue;
                    }
                    _ => return Err(self.err(self.i, format!("only \\$ \\{{ \\}} \\% \\& \\_ \\# may be used inside {what}"))),
                }
            }
            if c == '{' {
                return Err(self.err(self.i, format!("a {{ inside {what}")));
            }
            t.push(c);
            self.i += 1;
        }
        if self.i >= self.s.len() {
            return Err(self.err(open, "a { with no }"));
        }
        self.i += 1;
        Ok(t)
    }

    fn delimiter(&mut self, which: &str) -> Result<Option<char>, MathError> {
        self.skip_space();
        let at = self.i;
        if at >= self.s.len() {
            return Err(self.err(at.saturating_sub(1), format!("\\{which} needs a delimiter")));
        }
        let c = self.s[at];
        if c == '\\' {
            let name = self.command()?;
            return match name.as_str() {
                "langle" => Ok(Some('⟨')),
                "rangle" => Ok(Some('⟩')),
                "{" | "lbrace" => Ok(Some('{')),
                "}" | "rbrace" => Ok(Some('}')),
                "|" | "Vert" | "lVert" | "rVert" => Ok(Some('∥')),
                "vert" | "lvert" | "rvert" => Ok(Some('|')),
                "lfloor" => Ok(Some('⌊')),
                "rfloor" => Ok(Some('⌋')),
                "lceil" => Ok(Some('⌈')),
                "rceil" => Ok(Some('⌉')),
                _ => Err(self.err(at, format!("\\{name} is not a delimiter for \\{which}"))),
            };
        }
        self.i += 1;
        match c {
            '(' | ')' | '[' | ']' | '|' | '/' => Ok(Some(c)),
            '.' => Ok(None),
            _ => Err(self.err(at, format!("{c} is not a delimiter for \\{which}"))),
        }
    }

    fn atom(&mut self, style: Font) -> Result<Node, MathError> {
        let at = self.i;
        let c = self.s[at];
        if c == '{' {
            let v = self.argument(style)?;
            return Ok(Node::Group(v));
        }
        if c == '$' {
            return Err(self.err(at, "a $ inside math"));
        }
        if c == '&' || c == '#' || c == '%' || c == '~' {
            return Err(self.err(at, format!("{c} is not supported in a label (write \\{c} for the character)")));
        }
        if c != '\\' {
            self.i += 1;
            return Ok(char_node(c, style));
        }
        let name = self.command()?;
        let bad = |p: &Self, m: String| p.err(at, m);
        if let Some((ch, cls, font)) = symbol(&name) {
            let font = match (font, style) {
                (Font::Math, Font::Bold) => Font::MathBold,
                (Font::Main, Font::Bold) => Font::Bold,
                (f, _) => f,
            };
            return Ok(Node::Glyph { ch, font, cls });
        }
        if OPERATORS.contains(&name.as_str()) {
            return Ok(Node::Word { text: name, font: Font::Main, cls: Cls::Op });
        }
        if let Some(ch) = accent(&name) {
            let body = self.argument(style)?;
            return Ok(Node::Accent { ch, body });
        }
        let space = |mu: f64| Ok(Node::Space(mu * MU));
        match name.as_str() {
            "," => space(3.0),
            ":" | ">" => space(4.0),
            ";" => space(5.0),
            "!" => space(-3.0),
            " " => space(6.0),
            "quad" => space(18.0),
            "qquad" => space(36.0),
            "frac" | "dfrac" | "tfrac" => {
                let num = self.argument(style)?;
                self.skip_space();
                if self.i >= self.s.len() {
                    return Err(bad(self, format!("\\{name} needs two arguments")));
                }
                let den = self.argument(style)?;
                Ok(Node::Frac(num, den, name == "dfrac"))
            }
            "sqrt" => {
                self.skip_space();
                if self.i < self.s.len() && self.s[self.i] == '[' {
                    return Err(self.err(self.i, "\\sqrt[n] (a root index) is not supported"));
                }
                Ok(Node::Sqrt(self.argument(style)?))
            }
            "overline" => Ok(Node::Overline(self.argument(style)?)),
            "mathrm" => Ok(Node::Group(self.argument(Font::Main)?)),
            "mathbf" => Ok(Node::Group(self.argument(Font::Bold)?)),
            "mathit" => Ok(Node::Group(self.argument(Font::Italic)?)),
            "text" | "textrm" | "mbox" => Ok(Node::Word { text: self.raw_argument(&format!("\\{name}"))?, font: Font::Main, cls: Cls::Ord }),
            "operatorname" => Ok(Node::Word { text: self.raw_argument("\\operatorname")?, font: Font::Main, cls: Cls::Op }),
            "left" => {
                let left = self.delimiter("left")?;
                let body = self.list(style, false)?;
                if self.i >= self.s.len() || !self.peek_command("right") {
                    return Err(bad(self, "\\left with no \\right".into()));
                }
                self.i += 6;
                let right = self.delimiter("right")?;
                Ok(Node::Delimited { left, body, right })
            }
            "right" => Err(bad(self, "\\right with no \\left".into())),
            _ => Err(bad(self, format!("unknown command \\{name}"))),
        }
    }
}

// ---- layout -------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Kind {
    Glyph { text: String, font: Font, size: f64, w: f64 },
    Plain { text: String, w: f64 },
    Rule { w: f64, h: f64 },
    Surd { pts: Vec<(f64, f64)>, stroke: f64 },
}

#[derive(Clone, Debug)]
struct Item {
    kind: Kind,
    x: f64,
    /// down from the baseline; a rule's top edge
    y: f64,
}

#[derive(Clone, Debug, Default)]
struct HBox {
    items: Vec<Item>,
    w: f64,
    h: f64,
    d: f64,
}

impl HBox {
    fn put(&mut self, b: HBox, x: f64, y: f64) {
        for mut it in b.items {
            it.x += x;
            it.y += y;
            self.items.push(it);
        }
    }
}

/// [depth, height, italic, skew, width] of a glyph, in em.
fn metric(font: Font, ch: char) -> [f64; 5] {
    let ch = if ch == '≠' { '=' } else { ch };
    let table = FONTS[font as usize];
    if let Ok(k) = table.binary_search_by_key(&(ch as u32), |r| r.0) {
        return table[k].1;
    }
    // a face without the glyph: the regular face's metrics, then an estimate
    if font != Font::Main {
        let m = metric(Font::Main, ch);
        if m != FALLBACK {
            return m;
        }
    }
    FALLBACK
}

const FALLBACK: [f64; 5] = [0.0, 0.7, 0.0, 0.0, 0.6];

/// The TeX spacing between two atom classes, in mu; `tight` in scripts.
fn spacing(a: Cls, b: Cls, script: bool) -> f64 {
    use Cls::*;
    // (thin=3, medium=4, thick=5); the bool says the space is kept in scripts
    let (mu, keep) = match (a, b) {
        (Ord, Op) | (Op, Ord) | (Op, Op) | (Close, Op) | (Inner, Op) => (3.0, true),
        (Ord, Bin) | (Bin, Ord) | (Bin, Op) | (Bin, Open) | (Bin, Inner) | (Close, Bin) | (Inner, Bin) => (4.0, false),
        (Ord, Rel) | (Op, Rel) | (Rel, Ord) | (Rel, Op) | (Rel, Open) | (Rel, Inner) | (Close, Rel) | (Inner, Rel) => (5.0, false),
        (Ord, Inner) | (Op, Inner) | (Close, Inner) | (Inner, Ord) | (Inner, Open) | (Inner, Punct) | (Inner, Inner) => (3.0, false),
        (Punct, Rel) | (Punct, Ord) | (Punct, Op) | (Punct, Open) | (Punct, Close) | (Punct, Punct) | (Punct, Inner) => (3.0, false),
        _ => (0.0, true),
    };
    if script && !keep { 0.0 } else { mu }
}

/// TeX's rule for binary operators: a Bin with nothing to operate on is an Ord.
fn classes(nodes: &[Node]) -> Vec<Option<Cls>> {
    let mut out: Vec<Option<Cls>> = nodes.iter().map(|n| n.cls()).collect();
    let mut prev: Option<usize> = None;
    for k in 0..out.len() {
        let Some(c) = out[k] else { continue };
        if c == Cls::Bin {
            let before = prev.and_then(|p| out[p]);
            if matches!(before, None | Some(Cls::Bin) | Some(Cls::Op) | Some(Cls::Rel) | Some(Cls::Open) | Some(Cls::Punct)) {
                out[k] = Some(Cls::Ord);
            }
        }
        if matches!(c, Cls::Rel | Cls::Close | Cls::Punct) {
            if let Some(p) = prev {
                if out[p] == Some(Cls::Bin) {
                    out[p] = Some(Cls::Ord);
                }
            }
        }
        prev = Some(k);
    }
    if let Some(p) = prev {
        if out[p] == Some(Cls::Bin) {
            out[p] = Some(Cls::Ord);
        }
    }
    out
}

struct Lay {
    /// the math's text size in px
    base: f64,
}

impl Lay {
    fn size(&self, level: usize) -> f64 {
        self.base * SCALES[level.min(2)]
    }

    fn glyph(&self, ch: char, font: Font, size: f64) -> HBox {
        let m = metric(font, ch);
        let w = m[4] * size;
        HBox { items: vec![Item { kind: Kind::Glyph { text: ch.to_string(), font, size, w }, x: 0.0, y: 0.0 }], w, h: m[1] * size, d: m[0] * size }
    }

    fn list(&self, nodes: &[Node], level: usize) -> HBox {
        let size = self.size(level);
        let cls = classes(nodes);
        let mut b = HBox::default();
        let mut prev: Option<Cls> = None;
        for (n, c) in nodes.iter().zip(&cls) {
            if let (Some(p), Some(c)) = (prev, c) {
                b.w += spacing(p, *c, level > 0) * MU * size;
            }
            let nb = self.node(n, level);
            let x = b.w;
            b.h = b.h.max(nb.h);
            b.d = b.d.max(nb.d);
            b.w += nb.w;
            b.put(nb, x, 0.0);
            if c.is_some() {
                prev = *c;
            }
        }
        b
    }

    fn node(&self, n: &Node, level: usize) -> HBox {
        let size = self.size(level);
        match n {
            Node::Glyph { ch, font, .. } => self.glyph(*ch, *font, size),
            Node::Word { text, font, .. } => {
                let (mut w, mut h, mut d) = (0.0, 0.0f64, 0.0f64);
                for ch in text.chars() {
                    let m = metric(*font, ch);
                    w += m[4] * size;
                    h = h.max(m[1] * size);
                    d = d.max(m[0] * size);
                }
                HBox { items: vec![Item { kind: Kind::Glyph { text: text.clone(), font: *font, size, w }, x: 0.0, y: 0.0 }], w, h, d }
            }
            Node::Group(v) => self.list(v, level),
            Node::Space(em) => HBox { w: em * size, ..HBox::default() },
            Node::Scripts { base, sup, sub } => self.scripts(base, sup.as_deref(), sub.as_deref(), level),
            Node::Frac(num, den, display) => self.frac(num, den, *display, level),
            Node::Sqrt(body) => self.sqrt(body, level),
            Node::Overline(body) => {
                let bb = self.list(body, level);
                let th = RULE * size;
                let top = bb.h + 3.0 * th;
                let mut b = HBox { w: bb.w, h: top + th, d: bb.d, items: Vec::new() };
                b.items.push(Item { kind: Kind::Rule { w: bb.w, h: th }, x: 0.0, y: -(top + th) });
                b.put(bb, 0.0, 0.0);
                b
            }
            Node::Accent { ch, body } => self.accent(*ch, body, level),
            Node::Delimited { left, body, right } => self.delimited(*left, body, *right, level),
        }
    }

    fn scripts(&self, base: &Node, sup: Option<&[Node]>, sub: Option<&[Node]>, level: usize) -> HBox {
        let size = self.size(level);
        let ssize = self.size(level + 1);
        let bb = self.node(base, level);
        // a single character's scripts sit at fixed heights; a box's follow its top and bottom
        let (is_char, italic) = match base {
            Node::Glyph { ch, font, .. } if *font != Font::Size1 => (true, if matches!(font, Font::Math | Font::MathBold) { metric(*font, *ch)[2] * size } else { 0.0 }),
            Node::Glyph { ch, font, .. } => (false, metric(*font, *ch)[2] * size),
            _ => (false, 0.0),
        };
        let mut u = if is_char { 0.0 } else { bb.h - SUP_DROP * ssize };
        let mut v = if is_char { 0.0 } else { bb.d + SUB_DROP * ssize };
        let sp = sup.map(|s| self.list(s, level + 1));
        let sb = sub.map(|s| self.list(s, level + 1));
        let xh = X_HEIGHT * size;
        match (&sp, &sb) {
            (Some(p), None) => u = u.max(SUP1 * size).max(p.d + 0.25 * xh),
            (None, Some(q)) => v = v.max(SUB1 * size).max(q.h - 0.8 * xh),
            (Some(p), Some(q)) => {
                u = u.max(SUP1 * size).max(p.d + 0.25 * xh);
                v = v.max(SUB2 * size);
                let th = RULE * size;
                if (u - p.d) - (q.h - v) < 4.0 * th {
                    v = 4.0 * th - (u - p.d) + q.h;
                    let psi = 0.8 * xh - (u - p.d);
                    if psi > 0.0 {
                        u += psi;
                        v -= psi;
                    }
                }
            }
            (None, None) => {}
        }
        let mut b = HBox { w: bb.w, h: bb.h, d: bb.d, items: Vec::new() };
        let bw = bb.w;
        b.put(bb, 0.0, 0.0);
        let mut extra: f64 = 0.0;
        if let Some(p) = sp {
            b.h = b.h.max(u + p.h);
            extra = extra.max(italic + p.w);
            b.put(p, bw + italic, -u);
        }
        if let Some(q) = sb {
            b.d = b.d.max(v + q.d);
            extra = extra.max(q.w);
            b.put(q, bw, v);
        }
        b.w = bw + extra + SCRIPT_SPACE * size;
        b
    }

    fn frac(&self, num: &[Node], den: &[Node], display: bool, level: usize) -> HBox {
        let size = self.size(level);
        // text style sets the parts a size smaller; display style (\dfrac) at the same size
        let inner = if display { level } else { level + 1 };
        let n = self.list(num, inner);
        let dn = self.list(den, inner);
        let (th, a) = (RULE * size, AXIS * size);
        let (mut u, mut v) = if display { (NUM1 * size, DENOM1 * size) } else { (NUM2 * size, DENOM2 * size) };
        let phi = if display { 3.0 * th } else { th };
        let clr = (u - n.d) - (a + th / 2.0);
        if clr < phi {
            u += phi - clr;
        }
        let clr = (a - th / 2.0) - (dn.h - v);
        if clr < phi {
            v += phi - clr;
        }
        let pad = NULL_DELIM * size;
        let w = n.w.max(dn.w);
        let mut b = HBox { w: w + 2.0 * pad, h: u + n.h, d: v + dn.d, items: Vec::new() };
        let (nw, dw) = (n.w, dn.w);
        b.put(n, pad + (w - nw) / 2.0, -u);
        b.put(dn, pad + (w - dw) / 2.0, v);
        b.items.push(Item { kind: Kind::Rule { w, h: th }, x: pad, y: -(a + th / 2.0) });
        b
    }

    fn sqrt(&self, body: &[Node], level: usize) -> HBox {
        let size = self.size(level);
        let bb = self.list(body, level);
        let th = RULE * size;
        let gap = 0.1 * size;
        let top = bb.h.max(X_HEIGHT * size) + gap + th;
        let bottom = bb.d + 0.08 * size;
        let sw = 0.55 * size;
        let t = top + bottom;
        let y = |f: f64| -top + f * t;
        let end = sw + bb.w + 0.05 * size;
        let pts = vec![(0.0, y(0.62)), (0.16 * sw, y(0.55)), (0.45 * sw, bottom), (sw, -top + th / 2.0), (end, -top + th / 2.0)];
        let mut b = HBox { w: end + 0.05 * size, h: top, d: bottom, items: Vec::new() };
        b.items.push(Item { kind: Kind::Surd { pts, stroke: th * 1.25 }, x: 0.0, y: 0.0 });
        b.put(bb, sw, 0.0);
        b
    }

    fn accent(&self, ch: char, body: &[Node], level: usize) -> HBox {
        let size = self.size(level);
        let bb = self.list(body, level);
        let skew = match body {
            [Node::Glyph { ch, font, .. }] => metric(*font, *ch)[3] * size,
            _ => 0.0,
        };
        let mut b = HBox { w: bb.w, h: bb.h, d: bb.d, items: Vec::new() };
        let bh = bb.h;
        b.put(bb, 0.0, 0.0);
        if ch == '→' {
            // the arrow, at script size, just above the body
            let asz = 0.7 * size;
            let m = metric(Font::Main, ch);
            let lift = bh + 0.08 * size + m[0] * asz;
            let g = self.glyph(ch, Font::Main, asz);
            let aw = g.w;
            b.put(g, (b.w - aw) / 2.0 + skew, -lift);
            b.h = b.h.max(lift + m[1] * asz);
        }
        else {
            let m = metric(Font::Main, ch);
            let raise = (bh - X_HEIGHT * size).max(0.0);
            let g = self.glyph(ch, Font::Main, size);
            let aw = g.w;
            b.put(g, (b.w - aw) / 2.0 + skew, -raise);
            b.h = b.h.max(raise + m[1] * size);
        }
        b
    }

    fn delimited(&self, left: Option<char>, body: &[Node], right: Option<char>, level: usize) -> HBox {
        let size = self.size(level);
        let bb = self.list(body, level);
        let a = AXIS * size;
        let need = 2.0 * (bb.h - a).max(bb.d + a);
        let mut b = HBox::default();
        let delim = |b: &mut HBox, c: char| {
            let m = metric(Font::Main, c);
            let natural = (m[0] + m[1]) * size;
            let scale = if natural > 0.0 { (need / natural).max(1.0) } else { 1.0 };
            let ds = size * scale;
            let shift = (m[1] - m[0]) / 2.0 * ds - a;
            let g = self.glyph(c, Font::Main, ds);
            let x = b.w;
            b.w += g.w;
            b.h = b.h.max(m[1] * ds - shift);
            b.d = b.d.max(m[0] * ds + shift);
            b.put(g, x, shift);
        };
        match left {
            Some(c) => delim(&mut b, c),
            None => b.w += NULL_DELIM * size,
        }
        let x = b.w;
        b.w += bb.w;
        b.h = b.h.max(bb.h);
        b.d = b.d.max(bb.d);
        b.put(bb, x, 0.0);
        match right {
            Some(c) => delim(&mut b, c),
            None => b.w += NULL_DELIM * size,
        }
        b
    }
}

// ---- the label ------------------------------------------------------------------

/// A label laid out: the content of its `<text>`, the shapes beside it, and its size in px.
#[derive(Clone, Debug)]
pub struct Label {
    pub inner: String,
    shapes: Vec<Item>,
    pub w: f64,
    pub h: f64,
    pub d: f64,
    pub math: bool,
}

/// The text width of sans-serif text, estimated per character class (em).
pub fn sans_width(text: &str, size: f64) -> f64 {
    text.chars()
        .map(|c| match c {
            'i' | 'l' | 'j' | '.' | ',' | ':' | ';' | '\'' | '|' | '!' | ' ' => 0.27,
            'f' | 't' | 'r' | 'I' | '(' | ')' | '[' | ']' | '-' | '/' => 0.34,
            'm' | 'w' => 0.82,
            'M' | 'W' => 0.88,
            '0'..='9' => 0.57,
            c if c.is_ascii_uppercase() => 0.66,
            c if c.is_ascii_lowercase() => 0.53,
            _ => 0.6,
        })
        .sum::<f64>()
        * size
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// A number as the SVG writes it: two decimals, and never `-0.00`.
fn c(v: f64) -> String {
    let s = format!("{v:.2}");
    if s == "-0.00" { "0.00".to_string() } else { s }
}

/// Where the unescaped dollars are, by character index.
fn dollars(s: &[char]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut k = 0;
    while k < s.len() {
        if s[k] == '\\' && k + 1 < s.len() {
            k += 2;
            continue;
        }
        if s[k] == '$' {
            out.push(k);
        }
        k += 1;
    }
    out
}

/// Whether a label has math in it (an even, nonzero number of unescaped dollars).
pub fn has_math(s: &str) -> bool {
    let cs: Vec<char> = s.chars().collect();
    let n = dollars(&cs).len();
    n > 0 && n % 2 == 0
}

fn unescape_plain(s: &str) -> String {
    s.replace("\\$", "$")
}

/// A label drawn as its own text, math or not (after an error).
pub fn plain(s: &str, size: f64) -> Label {
    Label { inner: esc(s), shapes: Vec::new(), w: sans_width(s, size), h: 0.72 * size, d: 0.2 * size, math: false }
}

/// Check a label's math without laying it out.
pub fn check(s: &str) -> Result<(), MathError> {
    label(s, 12.0).map(|_| ())
}

/// Lay out a label at a text size in px. A label with no math gives its
/// escaped text (with `\$` as `$`), unchanged.
pub fn label(s: &str, size: f64) -> Result<Label, MathError> {
    let cs: Vec<char> = s.chars().collect();
    let ds = dollars(&cs);
    if ds.is_empty() || ds.len() % 2 == 1 {
        let t = if s.contains("\\$") { unescape_plain(s) } else { s.to_string() };
        return Ok(Label { inner: esc(&t), shapes: Vec::new(), w: sans_width(&t, size), h: 0.72 * size, d: 0.2 * size, math: false });
    }
    let lay = Lay { base: size * MATH_SCALE };
    let mut all = HBox { h: 0.72 * size, d: 0.2 * size, ..HBox::default() };
    let mut at = 0;
    let mut k = 0;
    while k <= ds.len() {
        let (end, next) = if k < ds.len() { (ds[k], ds[k + 1]) } else { (cs.len(), cs.len()) };
        let plain: String = cs[at..end].iter().collect();
        if !plain.is_empty() {
            let t = unescape_plain(&plain);
            let w = sans_width(&t, size);
            all.items.push(Item { kind: Kind::Plain { text: t, w }, x: all.w, y: 0.0 });
            all.w += w;
        }
        if k == ds.len() {
            break;
        }
        let mut p = Parser { s: &cs[end + 1..next], i: 0, base: end + 1 };
        let nodes = p.list(Font::Math, false)?;
        if p.i < p.s.len() {
            return Err(p.err(p.i, "\\right with no \\left"));
        }
        let b = lay.list(&nodes, 0);
        let x = all.w;
        all.w += b.w;
        all.h = all.h.max(b.h);
        all.d = all.d.max(b.d);
        all.put(b, x, 0.0);
        at = next + 1;
        k += 2;
    }
    Ok(Label { inner: tspans(&all.items), shapes: all.items.into_iter().filter(|i| matches!(i.kind, Kind::Rule { .. } | Kind::Surd { .. })).collect(), w: all.w, h: all.h, d: all.d, math: true })
}

fn font_attrs(f: Font) -> String {
    let (family, style, weight) = match f {
        Font::Main => (MAIN_STACK, "normal", "normal"),
        Font::Bold => (MAIN_STACK, "normal", "bold"),
        Font::Italic => (MAIN_STACK, "italic", "normal"),
        Font::Math => (MATH_STACK, "italic", "normal"),
        Font::MathBold => (MATH_STACK, "italic", "bold"),
        Font::Size1 => (SIZE1_STACK, "normal", "normal"),
    };
    format!(r#" font-family="{family}" font-style="{style}" font-weight="{weight}""#)
}

/// The runs as tspans: each moved from where the previous one ended.
fn tspans(items: &[Item]) -> String {
    let mut out = String::new();
    let (mut px, mut py) = (0.0, 0.0);
    // the open tspan: its font and size (None for plain text) and its text
    let mut open: Option<(Option<(Font, String)>, String, String)> = None;
    let flush = |out: &mut String, open: &mut Option<(Option<(Font, String)>, String, String)>| {
        if let Some((f, moves, text)) = open.take() {
            let attrs = match &f {
                Some((font, sz)) => format!(r#"{moves}{} font-size="{sz}""#, font_attrs(*font)),
                None => moves,
            };
            let _ = write!(out, "<tspan{attrs}>{}</tspan>", esc(&text));
        }
    };
    for it in items {
        let (text, w, f) = match &it.kind {
            Kind::Glyph { text, font, size, w } => (text, *w, Some((*font, c(*size)))),
            Kind::Plain { text, w } => (text, *w, None),
            _ => continue,
        };
        let dx = c(it.x - px);
        let dy = c(it.y - py);
        let moved = dx != "0.00" || dy != "0.00";
        let same = matches!(&open, Some((of, _, _)) if *of == f);
        if !moved && same {
            open.as_mut().unwrap().2.push_str(text);
        }
        else {
            flush(&mut out, &mut open);
            let mut moves = String::new();
            if dx != "0.00" {
                let _ = write!(moves, r#" dx="{dx}""#);
            }
            if dy != "0.00" {
                let _ = write!(moves, r#" dy="{dy}""#);
            }
            open = Some((f, moves, text.clone()));
        }
        px += dx.parse::<f64>().unwrap_or(0.0) + w;
        py += dy.parse::<f64>().unwrap_or(0.0);
    }
    flush(&mut out, &mut open);
    out
}

/// Where a label is anchored, as `text-anchor`.
#[derive(Clone, Copy, PartialEq)]
pub enum Anchor {
    Start,
    Middle,
    End,
}

impl Label {
    /// The fraction bars and radicals of a label whose text is at (x, y)
    /// with this anchor, turned by `rotate` degrees about that point.
    pub fn shapes_svg(&self, x: f64, y: f64, anchor: Anchor, rotate: f64, fill: &str) -> String {
        if self.shapes.is_empty() {
            return String::new();
        }
        let x0 = match anchor {
            Anchor::Start => 0.0,
            Anchor::Middle => -self.w / 2.0,
            Anchor::End => -self.w,
        };
        let turn = if rotate != 0.0 { format!(" rotate({})", c(rotate)) } else { String::new() };
        let mut s = format!(r#"<g transform="translate({} {}){turn}">"#, c(x), c(y));
        for it in &self.shapes {
            match &it.kind {
                Kind::Rule { w, h } => {
                    let _ = write!(s, r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{fill}"/>"#, c(x0 + it.x), c(it.y), c(*w), c(*h));
                }
                Kind::Surd { pts, stroke } => {
                    let mut d = String::new();
                    for (k, (px, py)) in pts.iter().enumerate() {
                        let _ = write!(d, "{}{} {}", if k == 0 { "M" } else { "L" }, c(x0 + it.x + px), c(it.y + py));
                    }
                    let _ = write!(s, r#"<path d="{d}" fill="none" stroke="{fill}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"/>"#, c(*stroke));
                }
                _ => {}
            }
        }
        s.push_str("</g>");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_labels_are_unchanged() {
        let l = label("E/A (MeV) & <x>", 12.0).unwrap();
        assert!(!l.math);
        assert_eq!(l.inner, "E/A (MeV) &amp; &lt;x&gt;");
        assert_eq!(label("price ($)", 12.0).unwrap().inner, "price ($)");
        assert_eq!(label(r"cost \$5", 12.0).unwrap().inner, "cost $5");
    }

    #[test]
    fn unknown_commands_are_located() {
        let e = label(r"energy $\foo x$", 12.0).unwrap_err();
        assert_eq!(e.msg, r"unknown command \foo");
        assert_eq!(e.col, 9);
        assert!(label(r"$x^$", 12.0).is_err());
        assert!(label(r"$\frac{a}$", 12.0).is_err());
        assert!(label(r"$\left( x$", 12.0).is_err());
        assert!(label(r"$x^2^3$", 12.0).is_err());
    }

    #[test]
    fn a_binary_minus_is_spaced_and_a_unary_one_is_not() {
        let unary = label("$-x$", 10.0).unwrap();
        let binary = label("$y-x$", 10.0).unwrap();
        let y = metric(Font::Math, 'y')[4] * 10.0 * MATH_SCALE;
        // binary: 4mu on each side
        assert!((binary.w - unary.w - y - 8.0 * MU * 10.0 * MATH_SCALE).abs() < 1e-9);
    }

    #[test]
    fn fractions_and_scripts_do_not_collide() {
        let f = label(r"$\frac{dE}{dt}$", 12.0).unwrap();
        assert_eq!(f.shapes.len(), 1);
        assert!(f.h > 0.72 * 12.0 && f.d > 0.2 * 12.0);
        let s = label("$10^{-14}$", 12.0).unwrap();
        assert!(s.inner.contains(r#"dy="-"#));
    }
}
