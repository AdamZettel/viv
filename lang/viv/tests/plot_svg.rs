// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! `viv plot` (src/figsvg.rs): the figure records of the corpus drawn as SVG,
//! compared byte for byte with the golden SVGs in `tests/figures/NAME.svg`.
//! The renderer promises the same bytes for the same records, so a change to
//! a golden file is a change to how every figure is drawn: review it as one.
//!
//!     VIV_WRITE_SVG=1 cargo test -p viv --test plot_svg     (re)write the golden SVGs

use std::path::Path;

fn corpus() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/figures")
}

/// The probes with a golden SVG: line, scatter, log axes, styles, marks and
/// ticks, bands and range bars, a grid with a right axis, equal aspect, a
/// diagram, heatmaps (settings, and marks in a window and in cells),
/// labels with math, and bars side by side under turned category labels
/// beside a dot plot with a row per y tick.
const GOLDEN: [&str; 11] = ["style", "log_axes", "marks", "intervals", "grid", "aspect", "diagram", "heatmap", "heatmap_marks", "mathtext", "bars"];

#[test]
fn the_corpus_draws_as_its_golden_svg() {
    let write = std::env::var_os("VIV_WRITE_SVG").is_some();
    for name in GOLDEN {
        let records = std::fs::read_to_string(corpus().join(format!("{name}.out"))).unwrap();
        let svg = viv::figsvg::render_page(&records, 640.0);
        let path = corpus().join(format!("{name}.svg"));
        if write {
            std::fs::write(&path, &svg).unwrap();
            continue;
        }
        let want = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("{} is missing (VIV_WRITE_SVG=1 writes it)", path.display()));
        assert!(svg == want, "{name}: the SVG differs from {}", path.display());
    }
}

#[test]
fn the_same_records_draw_the_same_bytes() {
    for name in GOLDEN {
        let records = std::fs::read_to_string(corpus().join(format!("{name}.out"))).unwrap();
        let a = viv::figsvg::render_page(&records, 640.0);
        let b = viv::figsvg::render_page(&records, 640.0);
        assert_eq!(a, b, "{name}");
        // no coordinate is nan or inf: a gap is left out, not drawn at nowhere
        assert!(!a.contains("NaN") && !a.contains("inf\""), "{name}: a non-finite value reached the drawing");
    }
}

#[test]
fn every_chart_is_its_own_document() {
    let records = std::fs::read_to_string(corpus().join("grid.out")).unwrap();
    let charts = viv::figsvg::render_all(&records, 640.0);
    // the grid (two panels) and the figure after it
    assert_eq!(charts.len(), 2);
    assert!(charts[0].contains("One swing, two panels") && charts[0].contains("(right axis)"));
    assert!(charts[1].contains("After the grid"));
}

/// Labels set as math (src/mathtext.rs), each laid out at 12px:
/// `tests/figures/mathtext_labels.tsv` holds the label, the content of its
/// `<text>`, and its shapes. www/scripts/mathtexttest.mjs holds the site's
/// renderer (www/src/lib/mathtext.js) to the same file.
const MATH_LABELS: [&str; 14] = [
    "$E/A$ (MeV)",
    r"$\lVert Q^\top Q - I \rVert$",
    r"$\kappa(A)$",
    "$10^{-14}$",
    r"$\omega/\omega_{pe}$",
    r"$\gamma(k)$",
    r"$\frac{dE}{dt}$",
    r"$\sqrt{\varepsilon}\,N$",
    "$x^2 - 5x + 6$",
    r"$\log_{10}\kappa$",
    r"$\dfrac{\partial f}{\partial x}$",
    r"$\hat{x}\cdot\vec{v} \le \left(\sum_i \alpha_i^2\right)^{1/2}$",
    r"$\mathrm{Re}\,\langle \psi | H | \psi \rangle \approx \hbar\omega$ (eV)",
    r"cost \$5, $\text{rate } r \propto e^{-t/\tau}$",
];

#[test]
fn math_labels_lay_out_as_their_golden() {
    let mut got = String::new();
    for l in MATH_LABELS {
        let m = viv::mathtext::label(l, 12.0).unwrap_or_else(|e| panic!("{l}: {e}"));
        assert!(m.math, "{l}: no math found");
        got.push_str(&format!("{l}\t{}\t{}\t{:.2}\n", m.inner, m.shapes_svg(0.0, 0.0, viv::mathtext::Anchor::Start, 0.0, "#000"), m.w));
    }
    let path = corpus().join("mathtext_labels.tsv");
    if std::env::var_os("VIV_WRITE_SVG").is_some() {
        std::fs::write(&path, &got).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("{} is missing (VIV_WRITE_SVG=1 writes it)", path.display()));
    for (g, w) in got.lines().zip(want.lines()) {
        assert_eq!(g, w);
    }
    assert_eq!(got.lines().count(), want.lines().count());
}

#[test]
fn a_math_error_is_reported_with_its_place() {
    let records = "figure: t ; $x$ ; $\\foo{y}$\nplot: a ; [0, 1] ; [0, 1]\n";
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_viv"))
        .args(["plot", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    std::io::Write::write_all(child.stdin.as_mut().unwrap(), records.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(err.contains(r#"line 1: in the label "$\foo{y}$": unknown command \foo at column 2"#), "{err}");
}
