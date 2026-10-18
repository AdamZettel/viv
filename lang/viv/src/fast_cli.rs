// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! Fast mode on the command line:
//!
//! - `viv build [--fast] [--release] <f.viv> [-o exe]`: the program as a
//!   native executable, compiled exact (`cc -O3`), release, or fast;
//! - `viv run --fast [--release] <f.viv>`: build fast and run it;
//! - `viv check --fast [--ulps N] [--wasm] [--json] [--show K] <f.viv>`: run
//!   the exact engines (the interpreter and the exact C build, and the direct
//!   wasm with `--wasm`), require them to agree byte for byte, run the fast
//!   build, and compare its output with theirs number by number
//!   (viv::fastcmp).
//!
//! The C is the same text in every mode; the mode is the compile line.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::Instant;

/// The ulp budget a number may differ by, before the condition numbers of
/// the run's linear algebra widen it.
pub const DEFAULT_ULPS: f64 = 1048576.0; // 2^20 ulps, 1.2e-10 relative

const UNIT_ROUNDOFF: f64 = 1.1102230246251565e-16; // 2^-53

#[derive(Clone, Copy, PartialEq)]
pub enum Mode {
    Exact,
    Release,
    Fast,
    FastRelease,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Exact => "exact",
            Mode::Release => "exact, release",
            Mode::Fast => "fast",
            Mode::FastRelease => "fast, release",
        }
    }
    fn fast(self) -> bool {
        matches!(self, Mode::Fast | Mode::FastRelease)
    }
    fn release(self) -> bool {
        matches!(self, Mode::Release | Mode::FastRelease)
    }
}

/// The flag that tunes for this machine's CPU, which `-O3` builds use.
fn native_cpu_flag() -> &'static str {
    if cfg!(target_arch = "x86_64") { "-march=native" } else { "-mcpu=native" }
}

/// The BLAS a fast build links, as compile flags, or none: Accelerate on
/// macOS; elsewhere what `VIV_BLAS_LIBS` names (`-lopenblas`). `VIV_BLAS=0`
/// turns it off.
pub fn blas_flags() -> Option<Vec<String>> {
    if std::env::var("VIV_BLAS").map_or(false, |v| v == "0") {
        return None;
    }
    if cfg!(target_os = "macos") {
        return Some(vec!["-DVIV_BLAS".into(), "-framework".into(), "Accelerate".into()]);
    }
    let libs = std::env::var("VIV_BLAS_LIBS").ok().filter(|s| !s.trim().is_empty())?;
    let mut v = vec!["-DVIV_BLAS".to_string()];
    v.extend(libs.split_whitespace().map(String::from));
    Some(v)
}

/// The freedoms a fast build leaves out (`--without contract,libm,blas`), to
/// tell which of them moves a number.
static WITHOUT: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();

fn without(freedom: &str) -> bool {
    WITHOUT.get().map_or(false, |w| w.iter().any(|x| x == freedom))
}

/// Read `--without a,b` from the arguments; false (and a message) for an
/// unknown freedom.
fn set_without(args: &[String]) -> bool {
    let list: Vec<String> = args.iter().position(|a| a == "--without").and_then(|k| args.get(k + 1)).map(|v| v.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()).unwrap_or_default();
    if let Some(bad) = list.iter().find(|x| !matches!(x.as_str(), "contract" | "libm" | "blas")) {
        eprintln!("error: --without {bad}: the freedoms are contract, libm and blas");
        return false;
    }
    let _ = WITHOUT.set(list);
    true
}

/// The C compiler's arguments for a mode, before the file names.
pub fn cc_flags(mode: Mode) -> Vec<String> {
    let mut f: Vec<String> = Vec::new();
    if mode == Mode::Exact {
        // -O3, not -O2: the exact semantics are the C text's (contraction
        // off by pragma, Viv's libm), not the optimizer's, and -O3 printed
        // the same bytes as -O2 on every program of the corpus.
        // No -mcpu=native: an exact
        // binary runs on any machine of its architecture.
        f.push("-O3".into());
        return f;
    }
    f.push("-O3".into());
    f.push(native_cpu_flag().into());
    if mode.release() {
        f.push("-DVIV_RELEASE".into());
    }
    if mode.fast() {
        f.push("-DVIV_FAST".into());
        if without("contract") {
            f.push("-DVIV_FAST_NO_CONTRACT".into());
            f.push("-ffp-contract=off".into());
        } else {
            // clang: contract everywhere except where a pragma says not to
            // (the shadow measurement's exact reference product)
            f.push(if cfg!(target_os = "macos") { "-ffp-contract=fast-honor-pragmas" } else { "-ffp-contract=fast" }.into());
        }
        if without("libm") {
            f.push("-DVIV_FAST_KEEP_LIBM".into());
        }
        // the generated runtime carries Viv's libm, unused in a fast build
        f.push("-Wno-unused-function".into());
        if !without("blas") {
            if let Some(b) = blas_flags() {
                f.extend(b);
            }
        }
    }
    f
}

/// What a fast build takes, in words, for the build line and the reports.
pub fn describe(mode: Mode) -> String {
    if !mode.fast() {
        return format!("{} (Viv's libm, no contraction{})", mode.name(), if mode.release() { ", no bounds checks" } else { "" });
    }
    let blas = match blas_flags() {
        _ if without("blas") => "no BLAS",
        Some(_) if cfg!(target_os = "macos") => "Accelerate for @ and lin_solve",
        Some(_) => "the BLAS for @ and lin_solve",
        None => "no BLAS",
    };
    let fma = if without("contract") { "no contraction" } else { "fused multiply-add" };
    let libm = if without("libm") { "Viv's libm" } else { "the platform libm" };
    format!("{} ({fma}, {libm}, {blas}{})", mode.name(), if mode.release() { ", no bounds checks" } else { "" })
}

/// Arguments to pass through to the compiler: `--use <f>` and `--set N=v`.
fn passthrough(args: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if (args[i] == "--use" || args[i] == "--set" || args[i] == "--using") && i + 1 < args.len() {
            out.push(args[i].clone());
            out.push(args[i + 1].clone());
            i += 1;
        }
        i += 1;
    }
    out
}

/// The program path: the first argument that is not a flag or a flag's value.
fn program_path(args: &[String], valued: &[&str]) -> Option<String> {
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if valued.contains(&a.as_str()) {
            i += 2;
            continue;
        }
        if !a.starts_with('-') {
            return Some(a.clone());
        }
        i += 1;
    }
    None
}

fn scratch(tag: &str) -> Result<PathBuf, String> {
    let d = std::env::temp_dir().join(format!("viv-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).map_err(|e| format!("error: cannot make {}: {e}", d.display()))?;
    Ok(d)
}

/// Write the program's C (by this same `viv`, so `#! use`, `--use` and
/// `--set` are handled as for `viv f.viv -o f.c`).
fn write_c(path: &str, extra: &[String], cfile: &Path) -> Result<(), String> {
    let me = std::env::current_exe().map_err(|e| format!("error: cannot find viv itself: {e}"))?;
    let out = Command::new(me).arg(path).arg("-o").arg(cfile).args(extra).output().map_err(|e| format!("error: {e}"))?;
    if !out.status.success() {
        return Err(format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)));
    }
    Ok(())
}

/// Compile a C file in a mode.
fn cc(cfile: &Path, exe: &Path, mode: Mode) -> Result<(), String> {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    // a program using the db_* builtins is built with SQLite (crate::db)
    let sqlite: Vec<String> = if std::fs::read_to_string(cfile).is_ok_and(|c| viv::db::uses_sqlite(&c)) {
        let mut v: Vec<String> = viv::db::SQLITE_FLAGS.iter().map(|f| f.to_string()).collect();
        v.push(viv::db::amalgamation_dir().join("sqlite3.c").display().to_string());
        v
    } else {
        Vec::new()
    };
    let out = Command::new(&cc).args(cc_flags(mode)).arg("-o").arg(exe).arg(cfile).args(&sqlite).arg("-lm").output().map_err(|e| format!("error: cannot run {cc}: {e}"))?;
    if !out.status.success() {
        return Err(format!("error: {cc} rejected the generated C ({}):\n{}", mode.name(), String::from_utf8_lossy(&out.stderr)));
    }
    Ok(())
}

/// How `viv build` compiles: `--single`
/// (or `VIV_SINGLE=1`) writes the whole program as one C file and compiles
/// it, as before the object cache; otherwise the C is split into units and
/// each is compiled through the cache, which `--no-cache` (or
/// `VIV_NO_CACHE=1`) bypasses. `--cache-clear` empties the cache first.
#[derive(Default, Clone, Copy)]
struct BuildOpts {
    single: bool,
    no_cache: bool,
}

static OPTS: std::sync::OnceLock<BuildOpts> = std::sync::OnceLock::new();

fn env_on(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.is_empty() && v != "0")
}

fn set_opts(args: &[String]) {
    let o = BuildOpts {
        single: args.iter().any(|a| a == "--single") || env_on("VIV_SINGLE"),
        no_cache: args.iter().any(|a| a == "--no-cache") || env_on("VIV_NO_CACHE"),
    };
    let _ = OPTS.set(o);
    if args.iter().any(|a| a == "--cache-clear") {
        let _ = std::fs::remove_dir_all(viv::objcache::cache_dir().join("obj"));
    }
}

/// What the last build did, for the line `viv build` prints.
static LAST: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

/// Build `path` in `mode` into `exe`, the C beside it in `dir`.
fn build_into(path: &str, extra: &[String], mode: Mode, dir: &Path, exe: &Path) -> Result<(), String> {
    let opts = OPTS.get().copied().unwrap_or_default();
    if opts.single {
        let cfile = dir.join("prog.c");
        write_c(path, extra, &cfile)?;
        *LAST.lock().unwrap() = "one translation unit".to_string();
        return cc(&cfile, exe, mode);
    }
    let sdir = dir.join("split");
    let me = std::env::current_exe().map_err(|e| format!("error: cannot find viv itself: {e}"))?;
    let out = Command::new(me).arg(path).arg("--split").arg(&sdir).args(extra).output().map_err(|e| format!("error: {e}"))?;
    if !out.status.success() {
        return Err(format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)));
    }
    compile_split(&sdir, mode, dir, exe, opts)
}

/// Compile the units of a split emission (`viv f.viv --split <sdir>`)
/// through the cache and link them into `exe`.
fn compile_split(sdir: &Path, mode: Mode, dir: &Path, exe: &Path, opts: BuildOpts) -> Result<(), String> {
    let list = std::fs::read_to_string(sdir.join("units.txt")).map_err(|e| format!("error: no unit list in {}: {e}", sdir.display()))?;
    let mut files: Vec<PathBuf> = list.lines().filter_map(|l| l.split('\t').next()).filter(|s| !s.is_empty()).map(|s| sdir.join("units").join(format!("{s}.c"))).collect();
    files.push(sdir.join("main.c"));
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let flags = cc_flags(mode);
    let t0 = Instant::now();
    let done = viv::objcache::compile_all(&cc, &flags, &files, &viv::objcache::cache_dir(), !opts.no_cache, dir, viv::objcache::jobs())?;
    let hits = done.iter().filter(|c| c.hit).count();
    let cpu: f64 = done.iter().map(|c| c.secs).sum();
    let objs: Vec<&PathBuf> = done.iter().map(|c| &c.object).collect();
    let out = Command::new(&cc).args(&flags).arg("-o").arg(exe).args(&objs).arg("-lm").output().map_err(|e| format!("error: cannot run {cc}: {e}"))?;
    if !out.status.success() {
        return Err(format!("error: {cc} could not link the units ({}):\n{}", mode.name(), String::from_utf8_lossy(&out.stderr)));
    }
    *LAST.lock().unwrap() = format!(
        "{} units, {} from the cache, {} compiled ({cpu:.1} s of cc), {:.1} s",
        done.len(),
        hits,
        done.len() - hits,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn mode_of(args: &[String]) -> Mode {
    let fast = args.iter().any(|a| a == "--fast");
    let release = args.iter().any(|a| a == "--release");
    match (fast, release) {
        (false, false) => Mode::Exact,
        (false, true) => Mode::Release,
        (true, false) => Mode::Fast,
        (true, true) => Mode::FastRelease,
    }
}

/// `viv build [--fast] [--release] [--use f] [--set N=v] <f.viv> [-o exe]`
pub fn build(args: &[String]) -> ExitCode {
    let rest = &args[2..];
    set_opts(rest);
    if !set_without(rest) {
        return ExitCode::FAILURE;
    }
    // `--cc-flags`: the C compiler's flags for the mode, for harnesses that
    // compile the C themselves (study/benchmarks, study/ports)
    if rest.iter().any(|a| a == "--cc-flags") {
        println!("{}", cc_flags(mode_of(rest)).join(" "));
        return ExitCode::SUCCESS;
    }
    let mode = mode_of(rest);
    // `--from-split <dir>`: the units a `viv f.viv --split <dir>` wrote,
    // perhaps edited since (a harness's hooks in main.c), compiled and linked
    if let Some(sdir) = rest.iter().position(|a| a == "--from-split").and_then(|k| rest.get(k + 1)).map(PathBuf::from) {
        let Some(exe) = rest.iter().position(|a| a == "-o").and_then(|k| rest.get(k + 1)).map(PathBuf::from) else {
            eprintln!("usage: viv build --from-split <dir> [--fast] [--release] -o <executable>");
            return ExitCode::FAILURE;
        };
        let r = scratch("build").and_then(|dir| {
            let r = compile_split(&sdir, mode, &dir, &exe, OPTS.get().copied().unwrap_or_default());
            let _ = std::fs::remove_dir_all(&dir);
            r
        });
        return match r {
            Ok(()) => {
                println!("built {}: {} [{}]", exe.display(), describe(mode), LAST.lock().unwrap());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{e}");
                ExitCode::FAILURE
            }
        };
    }
    let Some(path) = program_path(rest, &["-o", "--use", "--set", "--using", "--without"]) else {
        eprintln!("usage: viv build [--fast] [--release] [--single] [--no-cache] [--cache-clear] [--use <prelude.viv>] [--set NAME=value] <input.viv> [-o <executable>]\n       viv build --from-split <dir> [--fast] [--release] -o <executable>   (the units of `viv f.viv --split <dir>`)\n       viv build --cc-flags [--fast] [--release]   (the C compiler flags of the mode)");
        return ExitCode::FAILURE;
    };
    let exe = rest.iter().position(|a| a == "-o").and_then(|k| rest.get(k + 1)).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(&path).with_extension(""));
    let r = scratch("build").and_then(|dir| {
        let r = build_into(&path, &passthrough(rest), mode, &dir, &exe);
        let _ = std::fs::remove_dir_all(&dir);
        r
    });
    match r {
        Ok(()) => {
            println!("built {}: {} [{}]", exe.display(), describe(mode), LAST.lock().unwrap());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

/// `viv run --fast [--release] <f.viv>`: build fast, run, pass the output and
/// the exit status through.
pub fn run(args: &[String]) -> ExitCode {
    let rest = &args[2..];
    set_opts(rest);
    if !set_without(rest) {
        return ExitCode::FAILURE;
    }
    let Some(path) = program_path(rest, &["--use", "--set", "--using", "--without"]) else {
        eprintln!("usage: viv run --fast [--release] [--use <prelude.viv>] [--set NAME=value] <input.viv>");
        return ExitCode::FAILURE;
    };
    let mode = mode_of(rest);
    let dir = match scratch("run") {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let exe = dir.join("prog");
    let r = build_into(&path, &passthrough(rest), mode, &dir, &exe).and_then(|()| Command::new(&exe).status().map_err(|e| format!("error: cannot run the build: {e}")));
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(st) if st.success() => ExitCode::SUCCESS,
        Ok(_) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

/// A run's merged output (stdout, then stderr, as tools/three-engines.sh
/// takes it), without the interpreter's graph and trace lines and the
/// failure summaries, and its wall time.
fn capture(cmd: &mut Command) -> Result<(String, f64), String> {
    let t = Instant::now();
    let out = cmd.output().map_err(|e| format!("error: {e}"))?;
    let secs = t.elapsed().as_secs_f64();
    let mut text = String::from_utf8_lossy(&out.stdout).to_string();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    let kept: Vec<&str> = text
        .lines()
        .filter(|l| !l.starts_with("__graph__") && !l.starts_with("__trace__") && !l.ends_with("assertions FAILED") && !l.ends_with("assertion FAILED") && !l.ends_with("holds FAILED") && !l.ends_with("hold FAILED"))
        .collect();
    let mut s = kept.join("\n");
    if !s.is_empty() {
        s.push('\n');
    }
    Ok((s, secs))
}

/// The shadow file's lines (VIV_FAST_SHADOW): the largest forward-error bound
/// of a solve, and the gemm calls' worst share of their own bound.
struct Shadow {
    solves: usize,
    max_kappa: f64,
    max_solve_bound: f64,
    gemms: usize,
    gemm_worst: f64,
}

fn read_shadow(p: &Path) -> Shadow {
    let mut s = Shadow { solves: 0, max_kappa: 0.0, max_solve_bound: 0.0, gemms: 0, gemm_worst: 0.0 };
    let text = std::fs::read_to_string(p).unwrap_or_default();
    let field = |line: &str, key: &str| -> f64 {
        line.split_whitespace().find_map(|w| w.strip_prefix(key)).and_then(|v| v.parse::<f64>().ok()).unwrap_or(f64::INFINITY)
    };
    for line in text.lines() {
        if line.starts_with("solve ") {
            s.solves += 1;
            s.max_kappa = s.max_kappa.max(field(line, "kappa="));
            s.max_solve_bound = s.max_solve_bound.max(field(line, "bound="));
        } else if line.starts_with("gemm ") {
            s.gemms += 1;
            s.gemm_worst = s.gemm_worst.max(field(line, "of_bound="));
        }
    }
    s
}

/// `viv check --fast ...`
pub fn check(args: &[String]) -> ExitCode {
    let rest = &args[2..];
    set_opts(rest);
    if !set_without(rest) {
        return ExitCode::FAILURE;
    }
    const USAGE: &str = "usage: viv check --fast [--ulps N] [--without contract,libm,blas] [--release] [--wasm] [--no-interp] [--json] [--show K] [--use <prelude.viv>] [--set NAME=value] <input.viv>";
    let Some(path) = program_path(rest, &["--use", "--set", "--using", "--ulps", "--show", "--without"]) else {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    };
    let value = |flag: &str| rest.iter().position(|a| a == flag).and_then(|k| rest.get(k + 1)).and_then(|v| v.parse::<f64>().ok());
    let ulps = value("--ulps").unwrap_or(DEFAULT_ULPS);
    let show = value("--show").map(|v| v as usize).unwrap_or(8);
    let json = rest.iter().any(|a| a == "--json");
    let wasm = rest.iter().any(|a| a == "--wasm");
    // `--no-interp`: the exact C build alone as the reference, for a program
    // too long for the interpreter (tools/three-engines.sh is what shows the
    // interpreter and the C build agree)
    let interp = !rest.iter().any(|a| a == "--no-interp");
    let fast_mode = if rest.iter().any(|a| a == "--release") { Mode::FastRelease } else { Mode::Fast };
    let extra = passthrough(rest);
    let src = std::fs::read_to_string(&path).unwrap_or_default();
    if src.lines().any(|l| l.trim_end() == "#! timing") {
        println!("{path}: a timing program (#! timing): its output differs from run to run and is not compared");
        return ExitCode::SUCCESS;
    }
    let dir = match scratch("fastcheck") {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let result = check_in(&path, &extra, &dir, fast_mode, wasm, interp);
    let _ = std::fs::remove_dir_all(&dir);
    let c = match result {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    // the tolerance: the ulp budget, widened to the forward-error bound of the
    // worst-conditioned solve the fast run made
    let rtol = (ulps * UNIT_ROUNDOFF).max(c.shadow.max_solve_bound);
    let report = viv::fastcmp::compare(&c.exact, &c.fast, rtol);
    let ok = report.ok() && c.refs_agree.is_none();
    if json {
        println!(
            "{{\"program\":\"{}\",\"ok\":{},\"mode\":\"{}\",\"references\":[{}],\"references_agree\":{},\"rtol\":{:e},\"ulps\":{},\"solves\":{},\"max_kappa\":{},\"gemms\":{},\"gemm_worst_of_bound\":{},\"exact_s\":{:.4},\"fast_s\":{:.4},\"compare\":{}}}",
            path.replace('"', "\\\""),
            ok,
            fast_mode.name(),
            c.refs.iter().map(|r| format!("\"{r}\"")).collect::<Vec<_>>().join(","),
            c.refs_agree.is_none(),
            rtol,
            ulps,
            c.shadow.solves,
            if c.shadow.max_kappa.is_finite() { format!("{:e}", c.shadow.max_kappa) } else { "null".into() },
            c.shadow.gemms,
            if c.shadow.gemm_worst.is_finite() { format!("{:e}", c.shadow.gemm_worst) } else { "null".into() },
            c.exact_s,
            c.fast_s,
            viv::fastcmp::json(&report)
        );
    } else {
        println!("{path}: {} against the exact {}", if ok { "fast build within tolerance" } else { "FAST BUILD DIFFERS" }, c.refs.join(", "));
        if let Some(d) = &c.refs_agree {
            println!("  THE EXACT ENGINES DISAGREE ({d}): a bug in an engine, not rounding");
        }
        println!("  mode: {}", describe(fast_mode));
        println!("  tolerance: {:.3e} relative ({} ulps{})", rtol, ulps, if c.shadow.max_solve_bound > ulps * UNIT_ROUNDOFF { format!(", widened by a solve with condition number {:.2e}", c.shadow.max_kappa) } else { String::new() });
        if c.shadow.gemms + c.shadow.solves > 0 {
            println!("  BLAS/LAPACK calls: {} dgemm (worst {:.2e} of its own error bound), {} solves (largest condition number {:.2e})", c.shadow.gemms, c.shadow.gemm_worst, c.shadow.solves, c.shadow.max_kappa);
        }
        print!("  {}", viv::fastcmp::render(&report, show).replace("\n  ", "\n    "));
        println!("  time, one run each: exact native {:.3} s, fast {:.3} s", c.exact_s, c.fast_s);
    }
    if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}

struct Checked {
    exact: String,
    fast: String,
    refs: Vec<&'static str>,
    refs_agree: Option<String>,
    shadow: Shadow,
    exact_s: f64,
    fast_s: f64,
}

fn check_in(path: &str, extra: &[String], dir: &Path, fast_mode: Mode, wasm: bool, run_interp: bool) -> Result<Checked, String> {
    let me = std::env::current_exe().map_err(|e| format!("error: {e}"))?;
    // reference 1: the interpreter
    let interp = if run_interp { Some(capture(Command::new(&me).arg("run").args(extra).arg(path))?.0) } else { None };
    // reference 2: the exact C build
    let cfile = dir.join("prog.c");
    write_c(path, extra, &cfile)?;
    let exact_exe = dir.join("exact");
    cc(&cfile, &exact_exe, Mode::Exact)?;
    let (native, exact_s) = capture(&mut Command::new(&exact_exe))?;
    let mut refs = if interp.is_some() { vec!["interpreter", "C"] } else { vec!["C"] };
    let mut disagree = None;
    if interp.as_ref().map_or(false, |i| *i != native) {
        disagree = Some("interpreter vs C".to_string());
    }
    let interp = interp.unwrap_or_else(|| native.clone());
    // reference 3: the direct wasm, run in node
    if wasm {
        let wfile = dir.join("prog.wasm");
        let out = Command::new(&me).arg(path).arg("-o").arg(&wfile).args(extra).output().map_err(|e| format!("error: {e}"))?;
        if !out.status.success() {
            return Err(format!("error: the wasm build failed:\n{}", String::from_utf8_lossy(&out.stderr)));
        }
        let runner = std::env::var("VIV_RUN_WASM").map(PathBuf::from).unwrap_or_else(|_| repo_root().join("tools/run-wasm.mjs"));
        let (w, _) = capture(Command::new("node").arg(runner).arg(&wfile))?;
        refs.push("wasm");
        if w != interp && disagree.is_none() {
            disagree = Some("interpreter vs wasm".to_string());
        }
    }
    // the fast build, with the shadow measurements on
    let fast_exe = dir.join("fast");
    cc(&cfile, &fast_exe, fast_mode)?;
    let (fast, fast_s) = capture(&mut Command::new(&fast_exe))?;
    // a fast build's story says it is one;
    // the rest of the line is compared
    let fast = fast.replace("__story__ {\"mode\":\"fast\",", "__story__ {");
    // a second fast run measures each BLAS and LAPACK call against its own
    // error bound (VIV_FAST_SHADOW), when the program makes any
    let shadow_file = dir.join("shadow.txt");
    let c = std::fs::read_to_string(&cfile).unwrap_or_default();
    if c.contains("viv_mm_kernel(") || c.contains("viv_fast_solve(") {
        let (again, _) = capture(Command::new(&fast_exe).env("VIV_FAST_SHADOW", &shadow_file))?;
        if again.replace("__story__ {\"mode\":\"fast\",", "__story__ {") != fast {
            return Err("error: the fast build printed different output with VIV_FAST_SHADOW set".to_string());
        }
    }
    Ok(Checked { exact: interp, fast, refs, refs_agree: disagree, shadow: read_shadow(&shadow_file), exact_s, fast_s })
}

/// The repository root, for tools/run-wasm.mjs: VIV_ROOT, or the directory
/// above `target/` that holds this binary.
fn repo_root() -> PathBuf {
    if let Ok(r) = std::env::var("VIV_ROOT") {
        return PathBuf::from(r);
    }
    let me = std::env::current_exe().unwrap_or_default();
    me.ancestors().find(|p| p.join("tools/run-wasm.mjs").exists()).map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."))
}
