// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The runtime of compiled Viv: what a module from the direct wasm backend
//! (`crate::wasm_backend`) imports under "rt".
//!
//! It lives inside viv.wasm -- the compiler and interpreter built for the
//! browser -- and it is a child of the interpreter on purpose. A compiled
//! program does its own arithmetic, loops, calls and element access; every
//! builtin it cannot express in a few wasm instructions it hands to the
//! interpreter's own code, through `bridge_call`. So the text a program prints,
//! the order a sum is taken in, and the message a failed check stops with are
//! the interpreter's by construction, not by a second transcription that could
//! drift from the first. The one exception is where a copy through `Val` would
//! cost more than the work (element access, which the module does inline, and
//! the scalar math below); those call the same Rust functions the interpreter
//! calls, so they cannot drift either.
//!
//! Memory: the module imports this instance's memory, and every array lives in
//! it as a 32-byte header followed by its data:
//!
//! ```text
//!   +0  data pointer (always header + 32)     +12  dims[0..4], unused ones 0
//!   +4  element count                          +28  width: 1 real, 2 complex,
//!   +8  rank                                         3 function slots (i32), 0 raw bytes
//! ```
//!
//! Allocations are filed on an arena stack, the C runtime's scheme: a compiled
//! function that can allocate marks the height on entry and releases what is
//! above it on the way out, except the value it returns.
//!
//! Errors: the failing call records the interpreter's message and traps. The
//! host catches the trap, reads the message (`vrt_err_ptr`/`vrt_err_len`) and
//! prints `error: <message>`, which is what `viv run` prints. Every `RefCell`
//! borrow is released before the trap, so the next run starts clean
//! (`vrt_init` resets everything anyway).

use super::*;
use std::alloc::{alloc_zeroed, dealloc, Layout};

// Strings: their imports.
#[path = "wasm_str.rs"]
mod wasm_str;

/// Header size; data starts right after it.
const HDR: u32 = 32;

/// What a staged bridge argument is.
enum Arg {
    Val(Val),
    Str(String),
    /// An array argument: the Val handed to the interpreter, and the header its
    /// data is copied back into afterwards (the builtin may have written it).
    Arr(u32, Val),
}

struct Rt {
    interp: Rc<Interp>,
    arena: Vec<u32>,
    /// the string tables of this run (`vrt_static`), freed by the next `vrt_init`
    statics: Vec<(u32, usize)>,
    err: String,
    args: Vec<Arg>,
    /// the values a loop iteration keeps (`vrt_keep`, `vrt_release_kept`)
    keep: Vec<u32>,
    /// per builtin: calls through the bridge, and array elements copied
    /// across it (in and back) -- what `vrt_profile` reports
    profile: HashMap<String, (u64, u64)>,
}

thread_local! {
    static RT: RefCell<Option<Rt>> = const { RefCell::new(None) };
    /// Two doubles a complex-valued call leaves its result in.
    static SCRATCH: RefCell<Box<[f64; 2]>> = RefCell::new(Box::new([0.0; 2]));
    /// The maps of this run (interp/maps.rs). A module's map is a block on
    /// the arena (width `MAP_WIDTH`) holding an index into this table, so a
    /// map is freed by the arena's rules, like an array: when the def that
    /// made it returns without returning it, or at the end of the loop pass
    /// that made it. A freed map's slot is
    /// reused through the free list.
    static MAPS: RefCell<(Vec<Option<super::maps::Map>>, Vec<usize>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

fn interp() -> Rc<Interp> {
    RT.with(|r| r.borrow().as_ref().expect("vrt_init runs first").interp.clone())
}

// ---- memory ----------------------------------------------------------------

unsafe fn rd(p: u32) -> i32 {
    *(p as usize as *const i32)
}
unsafe fn wr(p: u32, v: i32) {
    *(p as usize as *mut i32) = v;
}
fn bytes_of(width: i32, size: i32) -> usize {
    let n = size.max(0) as usize;
    HDR as usize
        + match width {
            1 => 8 * n,
            2 => 16 * n,
            3 => 4 * n,
            _ => n,
        }
}
fn layout(bytes: usize) -> Layout {
    Layout::from_size_align(bytes.max(8), 8).expect("a valid layout")
}

/// Allocate a zeroed block with a header, and file it on the arena.
fn alloc(width: i32, shape: &[usize], size: usize) -> u32 {
    alloc_with(width, shape, size, true)
}

/// Allocate a block with a header and file it on the arena; its data is left
/// as the allocator hands it over unless `zeroed`. Only a caller that writes
/// every element may skip the zeroing, which for a big result of an
/// elementwise operation is a pass over memory saved.
fn alloc_with(width: i32, shape: &[usize], size: usize, zeroed: bool) -> u32 {
    let bytes = bytes_of(width, size as i32);
    let p = match recycled(bytes) {
        Some(p) => {
            if zeroed {
                unsafe { std::ptr::write_bytes(p as usize as *mut u8, 0, bytes) };
            }
            p
        }
        None => (unsafe { if zeroed { alloc_zeroed(layout(bytes)) } else { std::alloc::alloc(layout(bytes)) } }) as usize as u32,
    };
    if p == 0 {
        fail(format!("out of memory allocating {bytes} bytes"));
    }
    unsafe {
        wr(p, (p + HDR) as i32);
        wr(p + 4, size as i32);
        wr(p + 8, shape.len() as i32);
        for k in 0..4u32 {
            wr(p + 12 + 4 * k, shape.get(k as usize).map_or(0, |d| *d as i32));
        }
        wr(p + 28, width);
    }
    RT.with(|r| r.borrow_mut().as_mut().expect("vrt_init runs first").arena.push(p));
    p
}

unsafe fn free_block(p: u32) {
    if rd(p + 28) == MAP_WIDTH {
        // a map's block: its table slot goes too
        let k = rd(p + 12) as usize;
        MAPS.with(|m| {
            let mut m = m.borrow_mut();
            if let Some(slot) = m.0.get_mut(k) {
                *slot = None;
                m.1.push(k);
            }
        });
    }
    let bytes = bytes_of(rd(p + 28), rd(p + 4).max(grow_cap(p)));
    let cached = RECYCLE.with(|c| {
        let mut c = c.borrow_mut();
        if bytes < RECYCLE_MIN || c.1 + bytes > RECYCLE_BYTES {
            return false;
        }
        c.1 += bytes;
        c.0.entry(bytes).or_default().push(p);
        true
    });
    if !cached {
        dealloc(p as usize as *mut u8, layout(bytes));
    }
}

// Freed blocks, by size, for the next allocation of that size. A loop that
// updates an array allocates and frees blocks of one size, over and over, and
// dlmalloc took 47% of such a loop's time (pde/advection_1d); from the cache
// it is a pop. Small blocks go straight back to dlmalloc, which is quick with
// them, and the cache holds at most RECYCLE_BYTES.
const RECYCLE_MIN: usize = 1024;
const RECYCLE_BYTES: usize = 64 << 20;
thread_local! {
    static RECYCLE: RefCell<(HashMap<usize, Vec<u32>>, usize)> = RefCell::new((HashMap::new(), 0));
}

/// A block of `bytes` from the cache, if one is there.
fn recycled(bytes: usize) -> Option<u32> {
    if bytes < RECYCLE_MIN {
        return None;
    }
    RECYCLE.with(|c| {
        let mut c = c.borrow_mut();
        let p = c.0.get_mut(&bytes)?.pop()?;
        c.1 -= bytes;
        Some(p)
    })
}

/// Empty the cache into the allocator (a new run starts with it empty).
fn drain_recycled() {
    let all: Vec<(usize, Vec<u32>)> = RECYCLE.with(|c| {
        let mut c = c.borrow_mut();
        c.1 = 0;
        c.0.drain().collect()
    });
    for (bytes, ps) in all {
        for p in ps {
            unsafe { dealloc(p as usize as *mut u8, layout(bytes)) };
        }
    }
}

/// An array's shape, read from its header without allocating (the hot
/// builtins read two per call). Derefs to the slice of its dimensions, and
/// prints like one, so a shape-mismatch message reads as the interpreter's.
#[derive(Clone, Copy)]
struct Shape {
    dims: [usize; 4],
    rank: usize,
}
impl std::ops::Deref for Shape {
    type Target = [usize];
    fn deref(&self) -> &[usize] {
        &self.dims[..self.rank]
    }
}
impl PartialEq for Shape {
    fn eq(&self, other: &Shape) -> bool {
        **self == **other
    }
}
impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}

fn shape_of(p: u32) -> Shape {
    let rank = unsafe { rd(p + 8) }.clamp(0, 4) as usize;
    let mut dims = [0usize; 4];
    for (k, d) in dims.iter_mut().enumerate().take(rank) {
        *d = unsafe { rd(p + 12 + 4 * k as u32) } as usize;
    }
    Shape { dims, rank }
}

fn data_of<'a>(p: u32) -> &'a mut [f64] {
    unsafe {
        let n = rd(p + 4).max(0) as usize * if rd(p + 28) == 2 { 2 } else { 1 };
        std::slice::from_raw_parts_mut((p + HDR) as usize as *mut f64, n)
    }
}

/// A header's array as an interpreter value (a copy).
fn to_val(p: u32) -> Val {
    let shape = shape_of(p).to_vec();
    let data = data_of(p).to_vec();
    let a = Rc::new(RefCell::new(ArrData { shape, data }));
    if unsafe { rd(p + 28) } == 2 { Val::CArr(a) } else { Val::Arr(a) }
}

/// An interpreter array as a new header (a copy).
fn from_arr(a: &Arr, width: i32) -> u32 {
    let ab = a.borrow();
    let size: usize = ab.data.len() / width as usize;
    let p = alloc_with(width, &ab.shape, size, false);
    data_of(p).copy_from_slice(&ab.data);
    p
}

/// Record the interpreter's message and stop the program.
fn fail(msg: String) -> ! {
    // inside a test block, the test FAILs first (B344)
    let it = RT.with(|r| r.try_borrow().ok().and_then(|r| r.as_ref().map(|rt| rt.interp.clone())));
    if let Some(it) = it {
        it.test_stopped();
    }
    RT.with(|r| {
        if let Some(rt) = r.borrow_mut().as_mut() {
            rt.err = msg;
        }
    });
    core::arch::wasm32::unreachable()
}

fn string(p: u32, l: u32) -> String {
    let b = unsafe { std::slice::from_raw_parts(p as usize as *const u8, l as usize) };
    String::from_utf8_lossy(b).into_owned()
}

fn or_fail<T>(r: Result<T, String>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => fail(e),
    }
}

// ---- lifecycle and output --------------------------------------------------

/// Start a run: free what the last one left, and a fresh interpreter state
/// (test counters, the PRNG seed, step events, the story). `story` is 1 when
/// the program carries a debug lens, as `program_interp` decides it.
#[no_mangle]
pub extern "C" fn vrt_init(story: i32) {
    let old = RT.with(|r| r.borrow_mut().take());
    if let Some(old) = old {
        for p in old.arena {
            unsafe { free_block(p) };
        }
        for (p, bytes) in old.statics {
            unsafe { dealloc((p - HDR) as usize as *mut u8, layout(bytes)) };
        }
    }
    drain_recycled();
    MAPS.with(|m| *m.borrow_mut() = (Vec::new(), Vec::new()));
    DJ.with(|d| *d.borrow_mut() = None);
    // a new run's string table may sit where the last one's did
    EN_DESCS.with(|m| m.borrow_mut().clear());
    let interp = fresh_interp(false, HashMap::new(), HashMap::new(), HashMap::new());
    if story != 0 {
        *interp.story.borrow_mut() = Some(Story { buf: String::new(), total: 0, fails: 0 });
    }
    RT.with(|r| {
        *r.borrow_mut() = Some(Rt {
            interp: Rc::new(interp),
            arena: Vec::new(),
            statics: Vec::new(),
            err: String::new(),
            args: Vec::new(),
            keep: Vec::new(),
            profile: HashMap::new(),
        })
    });
}

/// The program's output so far, which the host reads and then clears.
#[no_mangle]
pub extern "C" fn vrt_out_ptr() -> u32 {
    interp().out.borrow().as_ptr() as usize as u32
}
#[no_mangle]
pub extern "C" fn vrt_out_len() -> u32 {
    interp().out.borrow().len() as u32
}
#[no_mangle]
pub extern "C" fn vrt_out_clear() {
    interp().out.borrow_mut().clear();
}
#[no_mangle]
pub extern "C" fn vrt_err_ptr() -> u32 {
    RT.with(|r| r.borrow().as_ref().map_or(0, |rt| rt.err.as_ptr() as usize as u32))
}
#[no_mangle]
pub extern "C" fn vrt_err_len() -> u32 {
    RT.with(|r| r.borrow().as_ref().map_or(0, |rt| rt.err.len() as u32))
}
/// The bridge's traffic this run, as JSON: `{"name": [calls, elements copied], ...}`,
/// most-copied first. `tools/run-wasm.mjs --profile` prints it.
#[no_mangle]
pub extern "C" fn vrt_profile() -> u64 {
    let mut rows: Vec<(String, (u64, u64))> = RT.with(|r| r.borrow().as_ref().map(|rt| rt.profile.clone().into_iter().collect()).unwrap_or_default());
    rows.sort_by(|a, b| (b.1 .1, b.1 .0).cmp(&(a.1 .1, a.1 .0)));
    let body: Vec<String> = rows.iter().map(|(n, (c, e))| format!("\"{n}\":[{c},{e}]")).collect();
    let s = format!("{{{}}}", body.join(","));
    let (p, l) = (s.as_ptr() as u64, s.len() as u64);
    std::mem::forget(s);
    (p << 32) | l
}

/// Failed assertions this run (the exit code's business, as in `viv run`).
#[no_mangle]
pub extern "C" fn vrt_fails() -> u32 {
    interp().fails.get()
}
/// Failed holds this run.
#[no_mangle]
pub extern "C" fn vrt_holds() -> u32 {
    interp().story.borrow().as_ref().map_or(0, |s| s.fails)
}
#[no_mangle]
pub extern "C" fn vrt_scratch() -> u32 {
    SCRATCH.with(|s| s.borrow().as_ptr() as usize as u32)
}
fn put_cplx(re: f64, im: f64) {
    SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        s[0] = re;
        s[1] = im;
    });
}

// ---- allocation --------------------------------------------------------------

#[no_mangle]
pub extern "C" fn vrt_mark() -> u32 {
    RT.with(|r| r.borrow().as_ref().expect("vrt_init runs first").arena.len() as u32)
}
/// Free everything filed since `mark` except `keep` (0: keep nothing).
#[no_mangle]
pub extern "C" fn vrt_release(mark: u32, keep: u32) {
    release_where(mark, |p, _| p == keep && keep != 0);
}

/// Free every allocation filed above `mark` that `kept` does not claim, in
/// place: a release runs once per call and per loop iteration, so it
/// allocates nothing itself.
fn release_where(mark: u32, kept: impl Fn(u32, &[u32]) -> bool) {
    RT.with(|r| {
        let mut r = r.borrow_mut();
        let rt = r.as_mut().expect("vrt_init runs first");
        let len = rt.arena.len();
        let mut w = (mark as usize).min(len);
        let now = EN_PASS_NOW.with(|c| c.get());
        for i in w..len {
            let p = rt.arena[i];
            // an enum node is kept when this pass's walk reached it, and
            // learns its new place
            let node = unsafe { rd(p + 28) } == EN_WIDTH;
            let keep = if node { (unsafe { rd(p + EN_PASS) }) == now } else { kept(p, &rt.keep) };
            if keep {
                if node {
                    unsafe { wr(p + HDR + 4, w as i32) };
                }
                rt.arena[w] = p;
                w += 1;
            } else {
                // free_block touches only the recycling cache and the allocator
                unsafe { free_block(p) };
            }
        }
        rt.arena.truncate(w);
        rt.keep.clear();
    });
    EN_PASS_NOW.with(|c| c.set(c.get().wrapping_add(1)));
}
/// The end of a loop iteration that can allocate: free what the iteration
/// filed, except what `vrt_keep` named -- the arrays and records the loop's
/// variables hold. Nothing else can reach an iteration's allocations once it
/// is over (an array holds numbers, not other arrays), so this frees exactly
/// the iteration's temporaries. The C runtime keeps them until the function
/// returns; the interpreter frees each when its last reference goes. A loop
/// of array updates -- `u = u - nu * (u - roll(u, 1.0))` -- would otherwise
/// grow memory by three arrays a step.
#[no_mangle]
pub extern "C" fn vrt_keep(p: u32) {
    RT.with(|r| r.borrow_mut().as_mut().expect("vrt_init runs first").keep.push(p));
}
#[no_mangle]
pub extern "C" fn vrt_release_kept(mark: u32) {
    release_where(mark, |p, keep| keep.contains(&p));
}

/// `disjoint` pins (crate::memory_lens): while a pinned `parallel for` runs,
/// which pass wrote each element of its pinned arrays (by block), the pass
/// running and the words naming the loop. The module runs the passes in
/// order, so a second pass writing an element is the overlap the pin rules
/// out: the run stops with the words the interpreter and the C runtime use.
struct DjRec {
    what: String,
    pass: i64,
    arrays: Vec<(u32, String, HashMap<usize, i64>)>,
}
thread_local! {
    static DJ: RefCell<Option<DjRec>> = const { RefCell::new(None) };
}
#[no_mangle]
pub extern "C" fn vrt_dj_begin(p: u32, l: u32) {
    let what = string(p, l);
    DJ.with(|d| *d.borrow_mut() = Some(DjRec { what, pass: -1, arrays: Vec::new() }));
}
#[no_mangle]
pub extern "C" fn vrt_dj_track(a: u32, p: u32, l: u32) {
    let name = string(p, l);
    DJ.with(|d| {
        if let Some(r) = d.borrow_mut().as_mut() {
            r.arrays.push((a, name, HashMap::new()));
        }
    });
}
#[no_mangle]
pub extern "C" fn vrt_dj_pass(i: f64) {
    DJ.with(|d| {
        if let Some(r) = d.borrow_mut().as_mut() {
            r.pass = i as i64;
        }
    });
}
/// An element write at `addr` (the block's address plus the element's offset,
/// as `Em::element_addr` leaves it).
#[no_mangle]
pub extern "C" fn vrt_dj_note(addr: u32) {
    let err = DJ.with(|d| {
        let mut d = d.borrow_mut();
        let r = d.as_mut()?;
        let pass = r.pass;
        for (a, name, owner) in r.arrays.iter_mut() {
            let n = elems(*a) as u32;
            if addr < *a || addr >= *a + 8 * n {
                continue;
            }
            let flat = ((addr - *a) / 8) as usize;
            match owner.get(&flat) {
                Some(&q) if q != pass => return Some(dj_error(q, pass, name, &shape_of(*a), flat, &r.what)),
                _ => {
                    owner.insert(flat, pass);
                }
            }
            return None;
        }
        None
    });
    if let Some(e) = err {
        fail(e);
    }
}
/// The address range the pinned arrays occupy (lowest block, end of the
/// highest one's elements), for the module's inline test before a write.
#[no_mangle]
pub extern "C" fn vrt_dj_lo() -> u32 {
    DJ.with(|d| d.borrow().as_ref().and_then(|r| r.arrays.iter().map(|(a, _, _)| *a).min()).unwrap_or(0))
}
#[no_mangle]
pub extern "C" fn vrt_dj_hi() -> u32 {
    DJ.with(|d| d.borrow().as_ref().and_then(|r| r.arrays.iter().map(|(a, _, _)| *a + 8 * elems(*a) as u32).max()).unwrap_or(0))
}
#[no_mangle]
pub extern "C" fn vrt_dj_end() {
    DJ.with(|d| *d.borrow_mut() = None);
}

/// A block of `bytes` raw bytes that lives for the run: the module's string
/// table. Not on the arena, so no release can take it.
#[no_mangle]
pub extern "C" fn vrt_static(bytes: u32) -> u32 {
    let size = bytes as usize + HDR as usize;
    let p = unsafe { alloc_zeroed(layout(size)) } as usize as u32;
    RT.with(|r| r.borrow_mut().as_mut().expect("vrt_init runs first").statics.push((p + HDR, size)));
    p + HDR
}
/// A raw zeroed block on the arena (a record), `bytes` long; returns its data.
#[no_mangle]
pub extern "C" fn vrt_block(bytes: u32) -> u32 {
    alloc(0, &[], bytes as usize) + HDR
}
/// `append(xs, r)` of an array of records: a new block of record pointers,
/// xs's and then `r`, which the module made a record of its own.
#[no_mangle]
pub extern "C" fn vrt_recs_push(a: u32, r: u32) -> u32 {
    let n = unsafe { rd(a + 12) }.max(0) as u32;
    let b = alloc(3, &[n as usize + 1], n as usize + 1);
    unsafe {
        for j in 0..n {
            wr(b + HDR + 4 * j, rd(a + HDR + 4 * j));
        }
        wr(b + HDR + 4 * n, r as i32);
    }
    b
}
/// A registry of `n` function slots (i32 table indices).
#[no_mangle]
pub extern "C" fn vrt_fnarr(n: u32) -> u32 {
    alloc(3, &[n as usize], n as usize)
}
/// An array literal's storage: `width` 1 or 2, rank 1 (`c` = -1) or 2.
#[no_mangle]
pub extern "C" fn vrt_lit(width: i32, r: u32, c: u32) -> u32 {
    // c is u32::MAX (-1) for a rank-1 literal of r elements
    if c == u32::MAX { alloc(width, &[r as usize], r as usize) } else { alloc(width, &[r as usize, c as usize], (r * c) as usize) }
}

/// `zeros` / `czeros` with the interpreter's arithmetic on the dimensions and
/// its element limit. `rank` of the four `d` are real.
fn zeros(width: i32, rank: u32, d: [f64; 4]) -> u32 {
    let (shape, total) = super::zeros_shape(&d[..rank as usize]);
    if width == 1 && total > MAX_ELEMS as u64 {
        fail(format!("zeros: array of {total} elements exceeds the {MAX_ELEMS}-element limit{LIMIT_OF}"));
    }
    if width == 2 && total > (MAX_ELEMS / 2) as u64 {
        fail(format!("czeros: complex array of {total} elements exceeds the {}-element limit{LIMIT_OF}", MAX_ELEMS / 2));
    }
    alloc(width, &shape, total as usize)
}
#[no_mangle]
pub extern "C" fn vrt_zeros(rank: u32, a: f64, b: f64, c: f64, d: f64) -> u32 {
    zeros(1, rank, [a, b, c, d])
}
#[no_mangle]
pub extern "C" fn vrt_czeros(rank: u32, a: f64, b: f64, c: f64, d: f64) -> u32 {
    zeros(2, rank, [a, b, c, d])
}

// ---- the checks the module does inline, stopping with the interpreter's words

#[no_mangle]
pub extern "C" fn vrt_oob(i: f64, k: u32, size: u32) {
    fail(format!("index {} out of bounds for dimension {k} (size {size})", i as i64))
}
/// An index that is not exactly an integer: the interpreter's words
/// (`interp::index_int`).
#[no_mangle]
pub extern "C" fn vrt_nonint(x: f64) {
    or_fail(super::index_int(x));
}
#[no_mangle]
pub extern "C" fn vrt_rank(rank: u32, n: u32) {
    fail(format!("array has rank {rank} but was indexed with {n} index/indices"))
}
/// A registry read out of range: the interpreter's message for any index.
#[no_mangle]
pub extern "C" fn vrt_fnoob(i: f64, size: u32) {
    fail(format!("index {} out of bounds for dimension 0 (size {size})", i as i64))
}
#[no_mangle]
pub extern "C" fn vrt_placeholder(p: u32, l: u32) {
    fail(format!("`{}` is a placeholder: it has a signature and no body yet", string(p, l)))
}

/// A check the parser inserted failed (ast `Lens::Stop`): stop with its words.
#[no_mangle]
pub extern "C" fn vrt_stop(p: u32, l: u32) {
    fail(string(p, l))
}

// ---- scalar math: the interpreter's own functions --------------------------

macro_rules! unary {
    ($($name:ident = $f:expr;)*) => { $(
        #[no_mangle]
        pub extern "C" fn $name(x: f64) -> f64 { ($f)(x) }
    )* };
}
macro_rules! binary {
    ($($name:ident = $f:expr;)*) => { $(
        #[no_mangle]
        pub extern "C" fn $name(x: f64, y: f64) -> f64 { ($f)(x, y) }
    )* };
}
unary! {
    vrt_exp = crate::vmath::exp; vrt_ln = crate::vmath::log; vrt_sin = crate::vmath::sin; vrt_cos = crate::vmath::cos; vrt_tan = crate::vmath::tan;
    vrt_asin = crate::vmath::asin; vrt_acos = crate::vmath::acos; vrt_atan = crate::vmath::atan;
    vrt_sinh = crate::vmath::sinh; vrt_cosh = crate::vmath::cosh; vrt_tanh = crate::vmath::tanh;
    vrt_round = f64::round; vrt_erf = erf_impl; vrt_gamma = gamma_impl;
    vrt_bnot = |a: f64| f64::from(!u32_of(a));
    vrt_u32 = |a: f64| f64::from(u32_of(a));
    vrt_i32 = |a: f64| f64::from(u32_of(a) as i32);
}
/// fma(a, b, c): IEEE 754's fusedMultiplyAdd, one rounding (compiler-builtins'
/// software fma on wasm32, which has no fused instruction)
#[no_mangle]
pub extern "C" fn vrt_fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}
binary! {
    vrt_pow = crate::vmath::pow; vrt_atan2 = crate::vmath::atan2; vrt_hypot = crate::vmath::hypot;
    vrt_fmod = |a: f64, b: f64| a % b; vrt_fmin = crate::vmath::fmin; vrt_fmax = crate::vmath::fmax;
    vrt_band = |a: f64, b: f64| f64::from(u32_of(a) & u32_of(b));
    vrt_bor = |a: f64, b: f64| f64::from(u32_of(a) | u32_of(b));
    vrt_bxor = |a: f64, b: f64| f64::from(u32_of(a) ^ u32_of(b));
    vrt_shl = |a: f64, k: f64| f64::from(shift_amount(k).map_or(0, |k| u32_of(a) << k));
    vrt_shr = |a: f64, k: f64| f64::from(shift_amount(k).map_or(0, |k| u32_of(a) >> k));
    vrt_floormod = |a: f64, b: f64| if b == 0.0 { fail(crate::interp::zero_divisor(crate::ast::BinOp::Mod)) } else { crate::interp::floor_mod(a, b) };
    vrt_floordiv = |a: f64, b: f64| if b == 0.0 { fail(crate::interp::zero_divisor(crate::ast::BinOp::FloorDiv)) } else { crate::interp::floor_div(a, b) };
    vrt_sar = |a: f64, k: f64| {
        let v = u32_of(a) as i32;
        f64::from(match shift_amount(k) { Some(k) => (v >> k) as u32, None => if v < 0 { u32::MAX } else { 0 } })
    };
}

/// `a / b` for complex scalars, by Smith's algorithm (`vmath::cdiv`, the
/// interpreter's); the quotient goes to the scratch.
#[no_mangle]
pub extern "C" fn vrt_cdiv(ar: f64, ai: f64, br: f64, bi: f64) {
    let (cr, ci) = crate::vmath::cdiv(ar, ai, br, bi);
    put_cplx(cr, ci);
}

/// A complex function of a complex scalar, by the interpreter's arm for it
/// (`exp`, `log`, `sqrt`, `sin`, `cos`, `tan`); the result goes to the scratch.
#[no_mangle]
pub extern "C" fn vrt_cfn(which: u32, re: f64, im: f64) {
    let name = ["exp", "log", "sqrt", "sin", "cos", "tan"][which as usize % 6];
    match bridge_val(name, vec![Arg::Val(Val::Cplx(re, im))]) {
        Val::Cplx(a, b) => put_cplx(a, b),
        _ => fail(format!("`{name}` of a complex number did not give one")),
    }
}

// ---- output ----------------------------------------------------------------

#[no_mangle]
pub extern "C" fn vrt_print(x: f64) {
    let it = interp();
    let mut out = it.out.borrow_mut();
    out.push_str(&format_g(x));
    out.push('\n');
}
#[no_mangle]
pub extern "C" fn vrt_show_num(p: u32, l: u32, x: f64) {
    interp().show(&string(p, l), &Val::Num(x));
}
#[no_mangle]
pub extern "C" fn vrt_show_bool(p: u32, l: u32, x: f64) {
    interp().show(&string(p, l), &Val::Bool(x != 0.0));
}
#[no_mangle]
pub extern "C" fn vrt_show_cplx(p: u32, l: u32, re: f64, im: f64) {
    interp().show(&string(p, l), &Val::Cplx(re, im));
}
#[no_mangle]
pub extern "C" fn vrt_show_arr(p: u32, l: u32, a: u32) {
    interp().show(&string(p, l), &to_val(a));
}
#[no_mangle]
pub extern "C" fn vrt_narrate(p: u32, l: u32) {
    let it = interp();
    let mut out = it.out.borrow_mut();
    out.push_str("narrate: ");
    out.push_str(&string(p, l));
    out.push('\n');
}
#[no_mangle]
pub extern "C" fn vrt_test_begin(p: u32, l: u32) {
    interp().test_begin(&string(p, l));
}
#[no_mangle]
pub extern "C" fn vrt_test_end() {
    interp().test_end();
}

// ---- steps and the story (the debug lens) ----------------------------------

/// One step event, `+def/id@depth` or `-def/id@depth`; into the story too
/// when one is being recorded, exactly as the interpreter's `step_event`.
#[no_mangle]
pub extern "C" fn vrt_step(enter: u32, dp: u32, dl: u32, ip: u32, il: u32, depth: u32) {
    interp().step_event(if enter != 0 { '+' } else { '-' }, &string(dp, dl), &string(ip, il), depth);
}
/// The end of main in a program with steps: the `__steps__` line, if any ran.
#[no_mangle]
pub extern "C" fn vrt_steps_flush() {
    let it = interp();
    if it.step_total.get() > 0 {
        let events = it.step_events.borrow().iter().map(|e| format!("\"{}\"", e)).collect::<Vec<_>>().join(",");
        it.out.borrow_mut().push_str(&format!("__steps__ {{\"total\":{},\"events\":[{}]}}\n", it.step_total.get(), events));
    }
}
/// `["call","def"]`, `["ret","def"]`, `["iter_end","def"]`.
#[no_mangle]
pub extern "C" fn vrt_story_mark(kp: u32, kl: u32, dp: u32, dl: u32) {
    let (kind, def) = (string(kp, kl), string(dp, dl));
    interp().story_event(|| format!("[\"{kind}\",\"{def}\"]"));
}
#[no_mangle]
pub extern "C" fn vrt_story_iter(dp: u32, dl: u32, n: f64) {
    let def = string(dp, dl);
    interp().story_event(|| format!("[\"iter\",\"{def}\",{}]", n as i64));
}
/// A watch: `vrt_story_watch_begin`, one `vrt_story_w_*` per name, then
/// `vrt_story_watch_end`, which records the event.
#[no_mangle]
pub extern "C" fn vrt_story_watch_begin() {
    RT.with(|r| r.borrow_mut().as_mut().expect("vrt_init runs first").args.clear());
}
#[no_mangle]
pub extern "C" fn vrt_story_w_num(np: u32, nl: u32, x: f64) {
    watch(string(np, nl), Val::Num(x));
}
#[no_mangle]
pub extern "C" fn vrt_story_w_bool(np: u32, nl: u32, x: f64) {
    watch(string(np, nl), Val::Bool(x != 0.0));
}
#[no_mangle]
pub extern "C" fn vrt_story_w_arr(np: u32, nl: u32, a: u32) {
    watch(string(np, nl), to_val(a));
}
fn watch(name: String, v: Val) {
    let text = or_fail(story_value(&name, &v));
    RT.with(|r| {
        let mut r = r.borrow_mut();
        let args = &mut r.as_mut().expect("vrt_init runs first").args;
        args.push(Arg::Str(format!("\"{name}\":{text}")));
    });
}
#[no_mangle]
pub extern "C" fn vrt_story_watch_end(dp: u32, dl: u32, ap: u32, al: u32) {
    let (def, at) = (string(dp, dl), string(ap, al));
    let fields: Vec<String> = RT.with(|r| {
        let mut r = r.borrow_mut();
        let args = &mut r.as_mut().expect("vrt_init runs first").args;
        args.drain(..).filter_map(|a| if let Arg::Str(s) = a { Some(s) } else { None }).collect()
    });
    interp().story_event(|| format!("[\"watch\",\"{def}\",\"{at}\",{{{}}}]", fields.join(",")));
}
/// The worst |difference| of a hold on arrays.
#[no_mangle]
pub extern "C" fn vrt_story_maxabs(a: u32) -> f64 {
    max_abs(data_of(a))
}
/// A hold: `ok`, the sentence (already JSON), and the worst difference when
/// it was a `within` check.
#[no_mangle]
pub extern "C" fn vrt_story_hold(dp: u32, dl: u32, ap: u32, al: u32, ok: u32, sp: u32, sl: u32, has_worst: u32, worst: f64) {
    let (def, at, sentence) = (string(dp, dl), string(ap, al), string(sp, sl));
    let it = interp();
    if ok == 0 {
        if let Some(st) = it.story.borrow_mut().as_mut() {
            st.fails += 1;
        }
    }
    let ok = ok != 0;
    it.story_event(|| {
        let w = if has_worst != 0 { story_num(worst) } else { "null".to_string() };
        format!("[\"hold\",\"{def}\",\"{at}\",{ok},{sentence},{w}]")
    });
}
/// The end of main in a program with a debug lens: the `__story__` line.
#[no_mangle]
pub extern "C" fn vrt_story_flush() {
    let it = interp();
    let line = it.story.borrow().as_ref().map(|st| format!("__story__ {{\"total\":{},\"events\":[{}]}}\n", st.total, st.buf));
    if let Some(line) = line {
        it.out.borrow_mut().push_str(&line);
    }
}

// ---- the bridge: any other builtin, by the interpreter's own arm -------------

fn stage(a: Arg) {
    RT.with(|r| r.borrow_mut().as_mut().expect("vrt_init runs first").args.push(a));
}
#[no_mangle]
pub extern "C" fn vrt_arg_num(x: f64) {
    stage(Arg::Val(Val::Num(x)));
}
#[no_mangle]
pub extern "C" fn vrt_arg_bool(x: f64) {
    stage(Arg::Val(Val::Bool(x != 0.0)));
}
#[no_mangle]
pub extern "C" fn vrt_arg_cplx(re: f64, im: f64) {
    stage(Arg::Val(Val::Cplx(re, im)));
}
/// An array argument. The same array passed twice is one interpreter value,
/// so a builtin that writes one argument and reads the other sees the write.
#[no_mangle]
pub extern "C" fn vrt_arg_arr(p: u32) {
    let seen = RT.with(|r| {
        r.borrow().as_ref().expect("vrt_init runs first").args.iter().find_map(|a| match a {
            Arg::Arr(q, v) if *q == p => Some(v.clone()),
            _ => None,
        })
    });
    stage(Arg::Arr(p, seen.unwrap_or_else(|| to_val(p))));
}
#[no_mangle]
pub extern "C" fn vrt_arg_str(p: u32, l: u32) {
    stage(Arg::Str(string(p, l)));
}

/// What a bridged builtin gave back: a value, or one of its own array
/// arguments (a builtin that returns an argument returns that array, not a
/// copy of it, and aliasing has to survive the trip).
enum Out {
    Val(Val),
    Alias(u32),
}

/// Run the interpreter's builtin `name` on the staged arguments. Array
/// arguments are copied back afterwards, since several builtins (`fill`,
/// `sort`, `fft`, the eigen-solvers' out-parameters) write through them.
fn bridge(name: &str, args: Vec<Arg>) -> Out {
    let it = interp();
    let span = knotkit::span::Span::synthetic(SourceId(0));
    let mut scope: HashMap<String, Val> = HashMap::new();
    let mut exprs = Vec::with_capacity(args.len());
    for (k, a) in args.iter().enumerate() {
        match a {
            Arg::Str(s) => exprs.push(Expr::Str { value: s.clone(), span }),
            Arg::Val(v) | Arg::Arr(_, v) => {
                let n = format!("a{k}");
                scope.insert(n.clone(), v.clone());
                exprs.push(Expr::Var { name: n, span });
            }
        }
    }
    let copied: usize = args.iter().map(|a| if let Arg::Arr(p, _) = a { 2 * data_of(*p).len() } else { 0 }).sum();
    RT.with(|r| {
        let mut r = r.borrow_mut();
        let e = r.as_mut().expect("vrt_init runs first").profile.entry(name.to_string()).or_insert((0, 0));
        e.0 += 1;
        e.1 += copied as u64;
    });
    let r = it.call(name, &exprs, &mut scope);
    drop(scope);
    write_back(&args);
    let v = or_fail(r);
    if let Val::Arr(a) | Val::CArr(a) = &v {
        for arg in &args {
            if let Arg::Arr(p, Val::Arr(b) | Val::CArr(b)) = arg {
                if Rc::ptr_eq(a, b) {
                    return Out::Alias(*p);
                }
            }
        }
    }
    Out::Val(v)
}

/// A bridged builtin that returns a value, never an array argument.
fn bridge_val(name: &str, args: Vec<Arg>) -> Val {
    match bridge(name, args) {
        Out::Val(v) => v,
        Out::Alias(_) => fail(format!("`{name}` returned an array where a value was expected")),
    }
}

fn write_back(args: &[Arg]) {
    for a in args {
        if let Arg::Arr(p, Val::Arr(v) | Val::CArr(v)) = a {
            let d = data_of(*p);
            let vb = v.borrow();
            if vb.data.len() == d.len() {
                d.copy_from_slice(&vb.data);
            }
        }
    }
}

fn take_args() -> Vec<Arg> {
    RT.with(|r| std::mem::take(&mut r.borrow_mut().as_mut().expect("vrt_init runs first").args))
}

/// A builtin whose result is a number or a boolean (as 1.0 / 0.0).
#[no_mangle]
pub extern "C" fn vrt_call_f(p: u32, l: u32) -> f64 {
    let name = string(p, l);
    match bridge_val(&name, take_args()) {
        Val::Num(x) => x,
        Val::Bool(b) => if b { 1.0 } else { 0.0 },
        _ => fail(format!("`{name}` did not return a number")),
    }
}
/// A builtin whose result is an array: a header (new, or the argument it is).
#[no_mangle]
pub extern "C" fn vrt_call_a(p: u32, l: u32) -> u32 {
    let name = string(p, l);
    let v = match bridge(&name, take_args()) {
        Out::Alias(h) => return h,
        Out::Val(v) => v,
    };
    match v {
        Val::Arr(a) => from_arr(&a, 1),
        Val::CArr(a) => from_arr(&a, 2),
        _ => fail(format!("`{name}` did not return an array")),
    }
}
/// A builtin whose result is a complex number, left in the scratch.
#[no_mangle]
pub extern "C" fn vrt_call_c(p: u32, l: u32) {
    let name = string(p, l);
    match bridge_val(&name, take_args()) {
        Val::Cplx(re, im) => put_cplx(re, im),
        Val::Num(x) => put_cplx(x, 0.0),
        _ => fail(format!("`{name}` did not return a complex number")),
    }
}

// ---- the hot builtins, without the copy -----------------------------------
//
// The bridge copies every array argument into an interpreter value and back.
// For a builtin that touches a small part of a big array (`slab` of one row,
// `shape`) or that runs in an inner loop, that copy is most of the cost -- in
// the LOBPCG post it was 99.7% of the elements the bridge moved. These do the
// work in place. Each is its interpreter arm (named beside it) over the
// module's memory: the same loop in the same order, the same checks with the
// same words. The corpus check (tools/three-engines.sh, tests/wasm_direct.rs)
// holds them to the interpreter byte for byte.

fn width(p: u32) -> usize {
    if unsafe { rd(p + 28) } == 2 { 2 } else { 1 }
}
fn elems(p: u32) -> usize {
    unsafe { rd(p + 4) }.max(0) as usize
}
/// A new array whose every element the caller then writes.
fn new_like(w: usize, shape: &[usize]) -> u32 {
    alloc_with(w as i32, shape, shape.iter().product(), false)
}
/// A new array of zeros (a sum accumulates into it).
fn new_zeros(w: usize, shape: &[usize]) -> u32 {
    alloc(w as i32, shape, shape.iter().product())
}

/// `slab(A, k)` (interp: "slab").
#[no_mangle]
pub extern "C" fn vrt_slab(a: u32, kf: f64) -> u32 {
    let shape = shape_of(a);
    if shape.len() < 2 {
        fail("`slab` needs an array of rank 2 or more".into());
    }
    let k = kf as usize;
    if k >= shape[0] {
        fail(format!("`slab` index {k} out of range (leading dimension {})", shape[0]));
    }
    let w = width(a);
    let len: usize = shape[1..].iter().product();
    let r = new_like(w, &shape[1..]);
    data_of(r).copy_from_slice(&data_of(a)[w * k * len..w * (k + 1) * len]);
    r
}

/// `setslab(A, k, S)` (interp: "setslab").
#[no_mangle]
pub extern "C" fn vrt_setslab(a: u32, kf: f64, s: u32) -> f64 {
    let k = kf as usize;
    let shape = shape_of(a);
    if shape.len() < 2 || k >= shape[0] {
        fail("`setslab`: index out of range, or the array has rank below 2".into());
    }
    let w = width(a);
    let len: usize = shape[1..].iter().product();
    let sb = data_of(s);
    if sb.len() != w * len {
        fail(format!("`setslab`: the slab has {} elements, the array's slabs have {len}", sb.len() / w));
    }
    // a slab of itself is the copy the interpreter takes first
    let src = sb.to_vec();
    data_of(a)[w * k * len..w * (k + 1) * len].copy_from_slice(&src);
    0.0
}

/// `take(A, idx)` (interp: "take").
#[no_mangle]
pub extern "C" fn vrt_take(a: u32, ix: u32) -> u32 {
    let w = width(a);
    let total = elems(a);
    let r = new_like(w, &shape_of(ix));
    let (src, idx, dst) = (data_of(a), data_of(ix), data_of(r));
    for (j, &f) in idx.iter().enumerate() {
        let k = f as usize;
        if f < 0.0 || k >= total {
            fail(format!("`take`: index {f} out of range (the array has {total} elements)"));
        }
        dst[w * j..w * (j + 1)].copy_from_slice(&src[w * k..w * (k + 1)]);
    }
    r
}

// Slices, copy and the joins: interp/slices.rs has the rules and the words;
// these run them over the module's memory.

fn slice_plan(a: u32, spec: u32) -> super::slices::Plan {
    match super::slices::plan(&shape_of(a), data_of(spec)) {
        Ok(p) => p,
        Err(e) => fail(e.msg),
    }
}
/// `a[...]` with a range (interp: "viv__slice_get").
#[no_mangle]
pub extern "C" fn vrt_getslice(a: u32, spec: u32) -> u32 {
    let p = slice_plan(a, spec);
    let w = width(a);
    let r = new_like(w, &p.shape);
    let (src, dst) = (data_of(a), data_of(r));
    for (j, &o) in p.offs.iter().enumerate() {
        dst[w * j..w * j + w].copy_from_slice(&src[w * o..w * o + w]);
    }
    r
}
/// `a[...] = v`, v an array (interp: "viv__slice_set").
#[no_mangle]
pub extern "C" fn vrt_setslice(a: u32, spec: u32, v: u32) -> f64 {
    let p = slice_plan(a, spec);
    let vs = shape_of(v);
    if vs[..] != p.shape[..] {
        fail(super::slices::shape_mismatch(&vs, &p.shape));
    }
    // the value may be the target: copied first, as the interpreter reads it
    let vals = data_of(v).to_vec();
    let w = width(a);
    let dst = data_of(a);
    for (j, &o) in p.offs.iter().enumerate() {
        dst[w * o..w * o + w].copy_from_slice(&vals[w * j..w * j + w]);
    }
    0.0
}
/// `a[...] = x`, x a number (interp: "viv__slice_set").
#[no_mangle]
pub extern "C" fn vrt_setslice_n(a: u32, spec: u32, x: f64) -> f64 {
    let p = slice_plan(a, spec);
    let dst = data_of(a);
    for &o in &p.offs {
        dst[o] = x;
    }
    0.0
}
/// `a[...] = z`, z a complex number (interp: "viv__slice_set").
#[no_mangle]
pub extern "C" fn vrt_setslice_c(a: u32, spec: u32, re: f64, im: f64) -> f64 {
    let p = slice_plan(a, spec);
    let dst = data_of(a);
    for &o in &p.offs {
        dst[2 * o] = re;
        dst[2 * o + 1] = im;
    }
    0.0
}
// ---- growable arrays (interp/growable.rs) --------------------------------------
//
// An array grown by `append` keeps two words in the header slots of dims[2]
// and dims[3], which a real array of rank 1 or 2 never uses (they are 0):
// +20 the capacity of its block, in elements, and +24 whether it is UNSEALED
// (1), which only an append's result is, until the program copies its
// handle anywhere (`viv__seal`, crate::growable). An append grows a block in
// place only when the compiler proved the variable its sole holder and the
// block is unsealed; otherwise it copies.

/// The capacity recorded in `p`'s header (0: the block holds exactly its elements).
unsafe fn grow_cap(p: u32) -> i32 {
    if rd(p + 8) <= 2 && rd(p + 28) == 1 { rd(p + 20).max(0) } else { 0 }
}
/// Room in `a` for `more` elements past its count, as the C runtime's
/// `viv_g_room`: the same block when it may grow in place and has room, the
/// block moved by a realloc when it may grow in place, else a new block with
/// a copy. The header's count and shape are the old ones; the caller sets them.
fn grow_room(a: u32, more: usize, own: f64) -> u32 {
    unsafe {
        let size = elems(a);
        let need = size + more;
        let cap = grow_cap(a) as usize;
        let unsealed = rd(a + 8) <= 2 && rd(a + 24) != 0;
        let inplace = own != 0.0 && unsealed;
        if inplace && cap >= need {
            return a;
        }
        // a copy is sized by the elements, not by the old block's capacity: a
        // sealed handle keeps its capacity word, and sizing every copy by it
        // doubled the block on each copy (B340: an array appended to in a def
        // and handed back reached 2^31 bytes within a few dozen calls)
        let have = if inplace && cap > 0 { cap } else { size };
        let newcap = (2 * have).max(need).max(8);
        let q = if inplace {
            let old = bytes_of(1, have as i32);
            let q = std::alloc::realloc(a as usize as *mut u8, layout(old), bytes_of(1, newcap as i32)) as usize as u32;
            if q == 0 {
                fail("out of memory growing an array".into());
            }
            RT.with(|r| {
                let mut r = r.borrow_mut();
                let arena = &mut r.as_mut().expect("vrt_init runs first").arena;
                match arena.iter().rposition(|&x| x == a) {
                    Some(i) => arena[i] = q,
                    None => arena.push(q),
                }
            });
            wr(q, (q + HDR) as i32);
            q
        } else {
            let q = alloc_with(1, &[newcap], newcap, false);
            // the old header, then the old elements
            std::ptr::copy_nonoverlapping((a + 4) as usize as *const u8, (q + 4) as usize as *mut u8, (HDR - 4) as usize);
            std::ptr::copy_nonoverlapping((a + HDR) as usize as *const u8, (q + HDR) as usize as *mut u8, 8 * size);
            q
        };
        wr(q + 20, newcap as i32);
        wr(q + 24, 1);
        q
    }
}
/// `viv__seal(a)`: its block may now be shared, so it never grows in place again.
#[no_mangle]
pub extern "C" fn vrt_seal(a: u32) -> u32 {
    unsafe {
        if rd(a + 8) <= 2 && rd(a + 28) == 1 {
            wr(a + 24, 0);
        }
    }
    a
}
fn grow_fail(callee: &str, a: u32, tail: super::growable::Tail) -> Vec<usize> {
    or_fail(super::growable::grown(callee, &shape_of(a), tail))
}
fn set_shape(p: u32, shape: &[usize]) {
    unsafe {
        wr(p + 4, shape.iter().product::<usize>() as i32);
        wr(p + 8, shape.len() as i32);
        for (k, d) in shape.iter().enumerate() {
            wr(p + 12 + 4 * k as u32, *d as i32);
        }
    }
}
/// `a = append(a, x)`, x a number (interp: "append").
#[no_mangle]
pub extern "C" fn vrt_append_n(a: u32, x: f64, own: f64) -> u32 {
    let shape = grow_fail("append", a, super::growable::Tail::Num);
    let n = elems(a);
    let q = grow_room(a, 1, own);
    set_shape(q, &shape);
    data_of(q)[n] = x;
    q
}
/// `a = append(a, n)` of an Int array: the same growth, the slot an i64.
#[no_mangle]
pub extern "C" fn vrt_iappend(a: u32, x: i64, own: f64) -> u32 {
    let n = elems(a);
    let q = grow_room(a, 1, own);
    set_shape(q, &[n + 1]);
    unsafe { *(((q + HDR) as usize as *mut i64).add(n)) = x };
    q
}
#[no_mangle]
pub extern "C" fn vrt_iseal(a: u32) -> u32 {
    vrt_seal(a)
}
/// An Int array's slots (B343).
fn ints_of<'a>(a: u32) -> &'a mut [i64] {
    unsafe { std::slice::from_raw_parts_mut((a + HDR) as usize as *mut i64, elems(a)) }
}
#[no_mangle]
pub extern "C" fn vrt_iextend(a: u32, b: u32, own: f64) -> u32 {
    grow_arr("extend", a, b, own)
}
#[no_mangle]
pub extern "C" fn vrt_igetslice(a: u32, spec: u32) -> u32 {
    vrt_getslice(a, spec)
}
#[no_mangle]
pub extern "C" fn vrt_isetslice(a: u32, spec: u32, v: u32) -> f64 {
    vrt_setslice(a, spec, v)
}
#[no_mangle]
pub extern "C" fn vrt_isum(a: u32) -> i64 {
    or_fail(super::ints::sum(ints_of(a)))
}
#[no_mangle]
pub extern "C" fn vrt_iminv(a: u32) -> i64 {
    or_fail(super::ints::extreme(ints_of(a), false))
}
#[no_mangle]
pub extern "C" fn vrt_imaxv(a: u32) -> i64 {
    or_fail(super::ints::extreme(ints_of(a), true))
}
#[no_mangle]
pub extern "C" fn vrt_isort(a: u32) -> f64 {
    ints_of(a).sort_unstable();
    0.0
}
/// `a = append(a, r)` (a row) or `a = extend(a, r)` (interp: those names).
fn grow_arr(callee: &str, a: u32, r: u32, own: f64) -> u32 {
    let rs = shape_of(r);
    let shape = grow_fail(callee, a, super::growable::Tail::Arr(&rs));
    let (n, k) = (elems(a), elems(r));
    let q = grow_room(a, k, own);
    // the row may be this array's own block, which the realloc may have moved
    let src = if r == a { q } else { r };
    unsafe { std::ptr::copy_nonoverlapping((src + HDR) as usize as *const f64, ((q + HDR) as usize as *mut f64).add(n), k) };
    set_shape(q, &shape);
    q
}
#[no_mangle]
pub extern "C" fn vrt_append_r(a: u32, r: u32, own: f64) -> u32 {
    grow_arr("append", a, r, own)
}
#[no_mangle]
pub extern "C" fn vrt_extend(a: u32, b: u32, own: f64) -> u32 {
    grow_arr("extend", a, b, own)
}

// ---- maps (interp/maps.rs: the rules are MapData's) ------------------------

use super::maps::{Col, MapData, KR, MV};

/// The width tag of a map's block on the arena: no elements, its table
/// index in `dims[0]`.
const MAP_WIDTH: i32 = 4;

fn map_of(h: u32) -> super::maps::Map {
    let k = unsafe { rd(h + 12) } as usize;
    MAPS.with(|m| m.borrow().0.get(k).cloned().flatten()).expect("a live map handle")
}
fn map_handle(m: MapData) -> u32 {
    let k = MAPS.with(|t| {
        let mut t = t.borrow_mut();
        let m = Some(Rc::new(RefCell::new(m)));
        match t.1.pop() {
            Some(k) => {
                t.0[k] = m;
                k
            }
            None => {
                t.0.push(m);
                t.0.len() - 1
            }
        }
    });
    let p = alloc(MAP_WIDTH, &[], 0);
    unsafe { wr(p + 12, k as i32) };
    p
}
fn map_list(data: Vec<f64>) -> u32 {
    let r = new_like(1, &[data.len()]);
    data_of(r).copy_from_slice(&data);
    r
}

/// A key or a value as a module passes it: a number (f64) or a string (an
/// i32 pointer into linear memory). A map keeps its own copy of a string
/// (the runtime's `Rc<Str>`), and a read hands out a new string on the
/// arena, as the C runtime does.
trait WasmMv: Copy {
    fn key<'a>(self) -> KR<'a>;
    fn owned(self) -> MV;
    fn out(v: MV) -> Self;
    /// a column as the module's value: an array of numbers or a list of strings
    fn list(c: Col) -> u32;
}
impl WasmMv for f64 {
    fn key<'a>(self) -> KR<'a> {
        KR::N(self)
    }
    fn owned(self) -> MV {
        MV::N(self)
    }
    fn out(v: MV) -> f64 {
        match v {
            MV::N(x) => x,
            MV::S(_) => unreachable!("a number map"),
        }
    }
    fn list(c: Col) -> u32 {
        match c {
            Col::N(v) => map_list(v),
            // an empty map's column has no kind yet
            Col::S(_) => map_list(Vec::new()),
        }
    }
}
impl WasmMv for u32 {
    fn key<'a>(self) -> KR<'a> {
        KR::S(wasm_str::str_of(self))
    }
    fn owned(self) -> MV {
        MV::S(super::strings::Str::new(wasm_str::str_of(self).to_string()))
    }
    fn out(v: MV) -> u32 {
        match v {
            MV::S(s) => wasm_str::new_str(&s.text),
            MV::N(_) => unreachable!("a map of strings"),
        }
    }
    fn list(c: Col) -> u32 {
        match c {
            Col::S(v) => wasm_str::list_of(&v.iter().map(|s| s.text.as_str()).collect::<Vec<_>>(), 0),
            Col::N(_) => wasm_str::list_of(&[], 0),
        }
    }
}

fn m_set<K: WasmMv, V: WasmMv>(h: u32, k: K, v: V) -> f64 {
    let r = map_of(h).borrow_mut().set(k.owned(), v.owned());
    or_fail(r);
    0.0
}
fn m_get<K: WasmMv, V: WasmMv>(h: u32, k: K) -> V {
    let r = map_of(h).borrow().get(k.key());
    V::out(r.unwrap_or_else(|| fail(super::maps::missing_key(k.key()))))
}
fn m_has<K: WasmMv>(h: u32, k: K) -> f64 {
    if map_of(h).borrow().has(k.key()) {
        1.0
    } else {
        0.0
    }
}
fn m_getd<K: WasmMv, V: WasmMv>(h: u32, k: K, d: V) -> V {
    let r = map_of(h).borrow().get(k.key());
    match r {
        Some(v) => V::out(v),
        None => d,
    }
}
fn m_remove<K: WasmMv, V: WasmMv>(h: u32, k: K) -> V {
    let r = map_of(h).borrow_mut().remove(k.key());
    V::out(r.unwrap_or_else(|| fail(super::maps::missing_key(k.key()))))
}
fn m_removed<K: WasmMv, V: WasmMv>(h: u32, k: K, d: V) -> V {
    let r = map_of(h).borrow_mut().remove(k.key());
    match r {
        Some(v) => V::out(v),
        None => d,
    }
}
fn m_len(h: u32) -> f64 {
    map_of(h).borrow().len() as f64
}
fn m_keys<K: WasmMv>(h: u32) -> u32 {
    let ks = map_of(h).borrow().keys();
    K::list(ks)
}
fn m_values<V: WasmMv>(h: u32) -> u32 {
    let vs = map_of(h).borrow().values();
    V::list(vs)
}
fn m_copy(h: u32) -> u32 {
    let m = map_of(h).borrow().copied();
    map_handle(m)
}
fn m_clear(h: u32) -> f64 {
    map_of(h).borrow_mut().clear();
    0.0
}

/// The runtime functions of one kind of map: `vrt_map_<op>` for numbers to
/// numbers, `vrt_map_<op>_<k><v>` for the others (maps::kinded_fns has the
/// list the backend imports).
macro_rules! map_fns {
    ($k:ty, $v:ty, $new:ident, $set:ident, $get:ident, $has:ident, $getd:ident, $remove:ident, $removed:ident, $len:ident, $keys:ident, $values:ident, $copy:ident, $clear:ident) => {
        #[no_mangle]
        pub extern "C" fn $new() -> u32 {
            map_handle(MapData::new())
        }
        #[no_mangle]
        pub extern "C" fn $set(h: u32, k: $k, v: $v) -> f64 {
            m_set(h, k, v)
        }
        #[no_mangle]
        pub extern "C" fn $get(h: u32, k: $k) -> $v {
            m_get::<$k, $v>(h, k)
        }
        #[no_mangle]
        pub extern "C" fn $has(h: u32, k: $k) -> f64 {
            m_has(h, k)
        }
        #[no_mangle]
        pub extern "C" fn $getd(h: u32, k: $k, d: $v) -> $v {
            m_getd(h, k, d)
        }
        #[no_mangle]
        pub extern "C" fn $remove(h: u32, k: $k) -> $v {
            m_remove::<$k, $v>(h, k)
        }
        #[no_mangle]
        pub extern "C" fn $removed(h: u32, k: $k, d: $v) -> $v {
            m_removed(h, k, d)
        }
        #[no_mangle]
        pub extern "C" fn $len(h: u32) -> f64 {
            m_len(h)
        }
        #[no_mangle]
        pub extern "C" fn $keys(h: u32) -> u32 {
            m_keys::<$k>(h)
        }
        #[no_mangle]
        pub extern "C" fn $values(h: u32) -> u32 {
            m_values::<$v>(h)
        }
        #[no_mangle]
        pub extern "C" fn $copy(h: u32) -> u32 {
            m_copy(h)
        }
        #[no_mangle]
        pub extern "C" fn $clear(h: u32) -> f64 {
            m_clear(h)
        }
    };
}

map_fns!(f64, f64, vrt_map_new, vrt_map_set, vrt_map_get, vrt_map_has, vrt_map_getd, vrt_map_remove, vrt_map_removed, vrt_map_len, vrt_map_keys, vrt_map_values, vrt_map_copy, vrt_map_clear);
map_fns!(u32, f64, vrt_map_new_sn, vrt_map_set_sn, vrt_map_get_sn, vrt_map_has_sn, vrt_map_getd_sn, vrt_map_remove_sn, vrt_map_removed_sn, vrt_map_len_sn, vrt_map_keys_sn, vrt_map_values_sn, vrt_map_copy_sn, vrt_map_clear_sn);
map_fns!(f64, u32, vrt_map_new_ns, vrt_map_set_ns, vrt_map_get_ns, vrt_map_has_ns, vrt_map_getd_ns, vrt_map_remove_ns, vrt_map_removed_ns, vrt_map_len_ns, vrt_map_keys_ns, vrt_map_values_ns, vrt_map_copy_ns, vrt_map_clear_ns);
map_fns!(u32, u32, vrt_map_new_ss, vrt_map_set_ss, vrt_map_get_ss, vrt_map_has_ss, vrt_map_getd_ss, vrt_map_remove_ss, vrt_map_removed_ss, vrt_map_len_ss, vrt_map_keys_ss, vrt_map_values_ss, vrt_map_copy_ss, vrt_map_clear_ss);

/// A map handed to the interpreter's arm (`print`): the same `Rc`, so the
/// interpreter reads the map the module changed.
#[no_mangle]
pub extern "C" fn vrt_arg_map(h: u32) {
    stage(Arg::Val(Val::Map(map_of(h))));
}

/// `copy(a)` (interp: "copy").
#[no_mangle]
pub extern "C" fn vrt_copy(a: u32) -> u32 {
    let r = new_like(width(a), &shape_of(a));
    data_of(r).copy_from_slice(data_of(a));
    r
}
fn cat_out(shape: &[usize], data: &[f64], w: usize) -> u32 {
    let r = new_like(w, shape);
    data_of(r).copy_from_slice(data);
    r
}
/// `concat`, `vcat` or `hcat` of two arrays (interp: those names, folded).
#[no_mangle]
pub extern "C" fn vrt_cat(mode: f64, a: u32, b: u32) -> u32 {
    let w = width(a);
    let (sa, sb) = (shape_of(a), shape_of(b));
    let (shape, data) = or_fail(super::slices::cat(mode as u8, (&sa, data_of(a)), Some((&sb, data_of(b))), w));
    cat_out(&shape, &data, w)
}
/// `concat`, `vcat` or `hcat` of one array.
#[no_mangle]
pub extern "C" fn vrt_cat1(mode: f64, a: u32) -> u32 {
    let w = width(a);
    let sa = shape_of(a);
    let (shape, data) = or_fail(super::slices::cat(mode as u8, (&sa, data_of(a)), None, w));
    cat_out(&shape, &data, w)
}

/// `put(A, idx, V)` (interp: "put").
#[no_mangle]
pub extern "C" fn vrt_put(a: u32, ix: u32, v: u32) -> f64 {
    let w = width(a);
    let (idx, vals) = (data_of(ix), data_of(v).to_vec());
    if vals.len() != w * idx.len() {
        fail("`put`: as many values as indices, please".into());
    }
    let total = elems(a);
    let dst = data_of(a);
    for (j, &f) in idx.iter().enumerate() {
        let k = f as usize;
        if f < 0.0 || k >= total {
            fail(format!("`put`: index {f} out of range (the array has {total} elements)"));
        }
        dst[w * k..w * (k + 1)].copy_from_slice(&vals[w * j..w * (j + 1)]);
    }
    0.0
}

/// `shape(A)` (interp: "shape").
#[no_mangle]
pub extern "C" fn vrt_shape(a: u32) -> u32 {
    let shape = shape_of(a);
    let r = new_like(1, &[shape.len()]);
    for (d, x) in shape.iter().zip(data_of(r).iter_mut()) {
        *x = *d as f64;
    }
    r
}

/// re / im / abs / conj over a complex array, `which` 0..3 (interp: the
/// complex-array arm of "re" | "im" | "conj" | "abs").
#[no_mangle]
pub extern "C" fn vrt_cmap(which: u32, a: u32) -> u32 {
    let shape = shape_of(a);
    let src = data_of(a);
    let n = src.len() / 2;
    if which == 3 {
        let r = new_like(2, &shape);
        let d = data_of(r);
        for k in 0..n {
            d[2 * k] = src[2 * k];
            d[2 * k + 1] = -src[2 * k + 1];
        }
        return r;
    }
    let r = new_like(1, &shape);
    let d = data_of(r);
    for k in 0..n {
        d[k] = match which {
            0 => src[2 * k],
            1 => src[2 * k + 1],
            _ => src[2 * k].vhypot(src[2 * k + 1]),
        };
    }
    r
}

/// A 1-argument math builtin over a real array, by the interpreter's table of
/// them (interp: "1-argument math builtins map elementwise over arrays").
/// `which` indexes `MAP_FNS`.
const MAP_FNS: [fn(f64) -> f64; 18] = [
    crate::vmath::exp, f64::sqrt, crate::vmath::sin, crate::vmath::cos, crate::vmath::tan, crate::vmath::log, f64::abs, f64::floor, f64::ceil,
    f64::round, crate::vmath::asin, crate::vmath::acos, crate::vmath::atan, crate::vmath::sinh, crate::vmath::cosh, crate::vmath::tanh, erf_impl, gamma_impl,
];
#[no_mangle]
pub extern "C" fn vrt_map(which: u32, a: u32) -> u32 {
    let f = MAP_FNS[which as usize];
    let r = new_like(1, &shape_of(a));
    for (d, &x) in data_of(r).iter_mut().zip(data_of(a).iter()) {
        *d = f(x);
    }
    r
}

/// `sum(a)` (interp: "sum"), in NumPy's order as the interpreter adds.
#[no_mangle]
pub extern "C" fn vrt_sum(a: u32) -> f64 {
    super::arrays::npsum(data_of(a))
}
/// `max(a)` / `min(a)` of one real array (interp: "max" | "min", one argument).
#[no_mangle]
pub extern "C" fn vrt_extreme(is_max: u32, a: u32) -> f64 {
    let d = data_of(a);
    if is_max != 0 { d.iter().fold(f64::NEG_INFINITY, |m, &x| crate::vmath::fmax(m, x)) } else { d.iter().fold(f64::INFINITY, |m, &x| crate::vmath::fmin(m, x)) }
}
/// `fill(a, v)`; a complex array fills with (re, im) (interp: "fill").
#[no_mangle]
pub extern "C" fn vrt_fill(a: u32, re: f64, im: f64) -> f64 {
    let d = data_of(a);
    if width(a) == 2 {
        for pair in d.chunks_mut(2) {
            pair[0] = re;
            pair[1] = im;
        }
    } else {
        for x in d.iter_mut() {
            *x = re;
        }
    }
    0.0
}

/// `-a` of a complex array: both parts negated (interp: `Neg` of a CArr).
#[no_mangle]
pub extern "C" fn vrt_cneg(a: u32) -> u32 {
    let x = data_of(a);
    let r = new_like(2, &shape_of(a));
    for (d, &v) in data_of(r).iter_mut().zip(x.iter()) {
        *d = -v;
    }
    r
}

/// `transpose(A)`, real or complex (interp: "transpose").
#[no_mangle]
pub extern "C" fn vrt_transpose(a: u32) -> u32 {
    let shape = shape_of(a);
    if shape.len() != 2 {
        // a real array of another rank: its axes reversed (interp/arrays.rs)
        if width(a) != 1 {
            fail("`transpose` takes a 2-D array".into());
        }
        let (s, d) = super::arrays::transpose_any(&shape, data_of(a));
        let r = new_like(1, &s);
        data_of(r).copy_from_slice(&d);
        return r;
    }
    let (rows, cols) = (shape[0], shape[1]);
    let w = width(a);
    let r = new_like(w, &[cols, rows]);
    let (src, dst) = (data_of(a), data_of(r));
    for i in 0..rows {
        for j in 0..cols {
            for c in 0..w {
                dst[w * (j * rows + i) + c] = src[w * (i * cols + j) + c];
            }
        }
    }
    r
}

/// `roll(a, k)` / `roll(a, k, axis)` (interp: "roll"); `has_axis` 0 for the
/// two-argument form.
#[no_mangle]
pub extern "C" fn vrt_roll(a: u32, k: f64, axisf: f64, has_axis: u32) -> u32 {
    let shape = shape_of(a);
    let rank = shape.len();
    let axis = if has_axis != 0 { axisf as usize } else { 0 };
    if has_axis == 0 && rank != 1 {
        fail("`roll(a, k)` is for a 1-D array; give the axis for a matrix: `roll(A, k, axis)`".into());
    }
    if axis >= rank {
        fail(format!("`roll`: axis {axis} out of range for a rank-{rank} array"));
    }
    let len = shape[axis];
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    let sh = (((k as i64) % len as i64) + len as i64) as usize % len;
    let r = new_like(1, &shape);
    let (src, dst) = (data_of(a), data_of(r));
    for o in 0..outer {
        for j in 0..len {
            let t = (j + sh) % len;
            for q in 0..inner {
                dst[(o * len + t) * inner + q] = src[(o * len + j) * inner + q];
            }
        }
    }
    r
}

/// `linspace(lo, hi, n)` (interp: "linspace").
#[no_mangle]
pub extern "C" fn vrt_linspace(lo: f64, hi: f64, nd: f64) -> u32 {
    if (nd as i64) < 0 {
        fail(format!("`linspace`: the number of points {} is negative", nd as i64));
    }
    let n = nd as usize;
    let h = if n > 1 { (hi - lo) / (n as f64 - 1.0) } else { 0.0 };
    let r = new_like(1, &[n]);
    let d = data_of(r);
    for (i, x) in d.iter_mut().enumerate() {
        *x = lo + h * i as f64;
    }
    if n > 1 {
        d[n - 1] = hi;
    }
    r
}

/// `complex(re, im)` over two real arrays (interp: "complex", the array arm).
#[no_mangle]
pub extern "C" fn vrt_carr_from(re: u32, im: u32) -> u32 {
    let shape = shape_of(re);
    if shape != shape_of(im) {
        fail("`complex(re, im)`: the two arrays must have the same shape".into());
    }
    let r = new_like(2, &shape);
    let (x, y, d) = (data_of(re), data_of(im), data_of(r));
    for k in 0..x.len() {
        d[2 * k] = x[k];
        d[2 * k + 1] = y[k];
    }
    r
}

/// The elementwise function for a `BinOp` tag: arithmetic, or a comparison
/// giving 1.0 / 0.0 (interp: `binary_array`).
fn real_op(op: u32) -> fn(f64, f64) -> f64 {
    match op {
        0 => |a, b| a + b,
        1 => |a, b| a - b,
        2 => |a, b| a * b,
        4 => |a, b| (a < b) as u8 as f64,
        5 => |a, b| (a <= b) as u8 as f64,
        6 => |a, b| (a > b) as u8 as f64,
        7 => |a, b| (a >= b) as u8 as f64,
        8 => |a, b| (a == b) as u8 as f64,
        9 => |a, b| (a != b) as u8 as f64,
        _ => |a, b| a / b,
    }
}

/// `a op b` elementwise on two real arrays of one shape; `op` is the
/// `BinOp` tag (interp: `binary_array`).
#[no_mangle]
pub extern "C" fn vrt_arith(op: u32, a: u32, b: u32) -> u32 {
    let (sa, sb) = (shape_of(a), shape_of(b));
    let f = real_op(op);
    if sa != sb {
        // NumPy's broadcasting (interp/arrays.rs), or the mismatch
        let (to, x, y) = or_fail(super::arrays::broadcast_pair(&sa, data_of(a), &sb, data_of(b), ""));
        let r = new_like(1, &to);
        for ((d, &x), &y) in data_of(r).iter_mut().zip(x.iter()).zip(y.iter()) {
            *d = f(x, y);
        }
        return r;
    }
    let r = new_like(1, &sa);
    for ((d, &x), &y) in data_of(r).iter_mut().zip(data_of(a).iter()).zip(data_of(b).iter()) {
        *d = f(x, y);
    }
    r
}
/// `a op s` (or `s op a` when `scalar_first`) with a scalar broadcast.
#[no_mangle]
pub extern "C" fn vrt_arith_s(op: u32, a: u32, s: f64, scalar_first: u32) -> u32 {
    let f = real_op(op);
    let r = new_like(1, &shape_of(a));
    let d = data_of(r);
    for (d, &x) in d.iter_mut().zip(data_of(a).iter()) {
        *d = if scalar_first != 0 { f(s, x) } else { f(x, s) };
    }
    r
}

/// `a @ b` for real arrays, batched (interp/matmul.rs).
#[no_mangle]
pub extern "C" fn vrt_matmul(a: u32, b: u32) -> u32 {
    let (sa, sb) = (shape_of(a), shape_of(b));
    let shape = super::matmul::shape(&sa, &sb).unwrap_or_else(|e| fail(e));
    let r = new_zeros(1, &shape);
    if let Err(e) = super::matmul::matmul_into(&sa, data_of(a), &sb, data_of(b), data_of(r)) {
        fail(e);
    }
    r
}

/// One complex operation on (ar, ai), (br, bi): the interpreter's formulas.
fn cop(op: u32, ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    match op {
        0 => (ar + br, ai + bi),
        1 => (ar - br, ai - bi),
        2 => crate::vmath::cmul(ar, ai, br, bi),
        _ => crate::vmath::cdiv(ar, ai, br, bi),
    }
}

/// Elementwise `a op b` with a complex array on at least one side and the
/// other a complex or real array of the same shape (interp: `binary_carray`).
#[no_mangle]
pub extern "C" fn vrt_carith(op: u32, a: u32, b: u32) -> u32 {
    let (sa, sb) = (shape_of(a), shape_of(b));
    if sa != sb {
        fail(format!("array shape mismatch: {sa:?} vs {sb:?}"));
    }
    if op == 3 {
        fail("a complex array can be divided by a scalar, not by an array".into());
    }
    let (wa, wb) = (width(a), width(b));
    let (x, y) = (data_of(a), data_of(b));
    let r = new_like(2, &sa);
    let d = data_of(r);
    let get = |v: &[f64], w: usize, k: usize| if w == 2 { (v[2 * k], v[2 * k + 1]) } else { (v[k], 0.0) };
    for k in 0..d.len() / 2 {
        let ((ar, ai), (br, bi)) = (get(x, wa, k), get(y, wb, k));
        let (cr, ci) = cop(op, ar, ai, br, bi);
        d[2 * k] = cr;
        d[2 * k + 1] = ci;
    }
    r
}
/// `a op s` (`s op a` when `scalar_first`), a a complex array, s a scalar.
#[no_mangle]
pub extern "C" fn vrt_carith_s(op: u32, a: u32, sr: f64, si: f64, scalar_first: u32) -> u32 {
    let x = data_of(a);
    let r = new_like(2, &shape_of(a));
    let d = data_of(r);
    for k in 0..d.len() / 2 {
        let (cr, ci) = if scalar_first != 0 { cop(op, sr, si, x[2 * k], x[2 * k + 1]) } else { cop(op, x[2 * k], x[2 * k + 1], sr, si) };
        d[2 * k] = cr;
        d[2 * k + 1] = ci;
    }
    r
}

/// `a @ b` for complex arrays (interp: `binary_carray`, the `@` arm).
#[no_mangle]
pub extern "C" fn vrt_cmatmul(a: u32, b: u32) -> u32 {
    let (sa, sb) = (shape_of(a), shape_of(b));
    let (ra, rb) = (sa.len(), sb.len());
    if ra == 1 && rb == 1 {
        fail(super::matmul::VECTORS.into());
    }
    if !((ra == 2 || ra == 1) && (rb == 2 || rb == 1)) {
        fail("`@` needs arrays (2-D, or a 1-D vector operand)".into());
    }
    let (m, k) = if ra == 2 { (sa[0], sa[1]) } else { (1, sa[0]) };
    let (k2, n) = if rb == 2 { (sb[0], sb[1]) } else { (sb[0], 1) };
    if k != k2 {
        fail(format!("`@` shape mismatch: {m}x{k} @ {k2}x{n}"));
    }
    if m.saturating_mul(n) > MAX_ELEMS / 2 {
        fail(format!("`@`: result of {m}x{n} exceeds the element limit"));
    }
    let r = if ra == 1 || rb == 1 { new_zeros(2, &[m * n]) } else { new_zeros(2, &[m, n]) };
    let (ad, bd, data) = (data_of(a), data_of(b), data_of(r));
    for i in 0..m {
        for p in 0..k {
            let (ar, ai) = (ad[2 * (i * k + p)], ad[2 * (i * k + p) + 1]);
            if ar == 0.0 && ai == 0.0 {
                continue;
            }
            // the interpreter's j loop, over row slices so the inner loop
            // carries no index checks; same sums in the same order
            let brow = &bd[2 * p * n..2 * (p * n + n)];
            let rrow = &mut data[2 * i * n..2 * (i * n + n)];
            for (rv, bv) in rrow.chunks_exact_mut(2).zip(brow.chunks_exact(2)) {
                let (br, bi) = (bv[0], bv[1]);
                rv[0] += ar * br - ai * bi;
                rv[1] += ar * bi + ai * br;
            }
        }
    }
    r
}

/// Elementwise max / min of two real arrays, or of one and a scalar (interp:
/// "min" | "max" with an array operand).
#[no_mangle]
pub extern "C" fn vrt_minmax(is_max: u32, a: u32, b: u32) -> u32 {
    let (sa, sb) = (shape_of(a), shape_of(b));
    let name = if is_max != 0 { "max" } else { "min" };
    if sa != sb {
        let (to, x, y) = or_fail(super::arrays::broadcast_pair(&sa, data_of(a), &sb, data_of(b), &format!("`{name}`: ")));
        let r = new_like(1, &to);
        for ((d, &p), &q) in data_of(r).iter_mut().zip(x.iter()).zip(y.iter()) {
            *d = if is_max != 0 { crate::vmath::fmax(p, q) } else { crate::vmath::fmin(p, q) };
        }
        return r;
    }
    let r = new_like(1, &sa);
    for ((d, &p), &q) in data_of(r).iter_mut().zip(data_of(a).iter()).zip(data_of(b).iter()) {
        *d = if is_max != 0 { crate::vmath::fmax(p, q) } else { crate::vmath::fmin(p, q) };
    }
    r
}
#[no_mangle]
pub extern "C" fn vrt_minmax_s(is_max: u32, a: u32, s: f64) -> u32 {
    let r = new_like(1, &shape_of(a));
    for (d, &p) in data_of(r).iter_mut().zip(data_of(a).iter()) {
        *d = if is_max != 0 { crate::vmath::fmax(p, s) } else { crate::vmath::fmin(p, s) };
    }
    r
}
/// `pow(a, s)` over a real array (interp: "pow" with an array).
#[no_mangle]
pub extern "C" fn vrt_pows(a: u32, s: f64) -> u32 {
    let r = new_like(1, &shape_of(a));
    for (d, &p) in data_of(r).iter_mut().zip(data_of(a).iter()) {
        *d = p.vpow(s);
    }
    r
}

// ---- Ints: the rules of interp/ints.rs, by import ----------

#[no_mangle]
pub extern "C" fn vrt_i_add(a: i64, b: i64) -> i64 {
    or_fail(ints::add(a, b))
}
#[no_mangle]
pub extern "C" fn vrt_i_sub(a: i64, b: i64) -> i64 {
    or_fail(ints::sub(a, b))
}
#[no_mangle]
pub extern "C" fn vrt_i_mul(a: i64, b: i64) -> i64 {
    or_fail(ints::mul(a, b))
}
#[no_mangle]
pub extern "C" fn vrt_i_fdiv(a: i64, b: i64) -> i64 {
    or_fail(ints::floordiv(a, b))
}
#[no_mangle]
pub extern "C" fn vrt_i_cdiv(a: i64, b: i64) -> i64 {
    or_fail(ints::ceildiv(a, b))
}
#[no_mangle]
pub extern "C" fn vrt_i_fmod(a: i64, b: i64) -> i64 {
    or_fail(ints::floormod(a, b))
}
#[no_mangle]
pub extern "C" fn vrt_i_pow(a: i64, k: i64) -> i64 {
    or_fail(ints::pow(a, k))
}
#[no_mangle]
pub extern "C" fn vrt_i_neg(a: i64) -> i64 {
    or_fail(ints::neg(a))
}
#[no_mangle]
pub extern "C" fn vrt_i_abs(a: i64) -> i64 {
    or_fail(ints::abs(a))
}
#[no_mangle]
pub extern "C" fn vrt_i_num(a: i64) -> f64 {
    or_fail(ints::to_num(a))
}
#[no_mangle]
pub extern "C" fn vrt_i_of(x: f64) -> i64 {
    or_fail(ints::of_num(x))
}
#[no_mangle]
pub extern "C" fn vrt_i_cmp(a: i64, x: f64) -> f64 {
    ints::cmp_num(a, x)
}
#[no_mangle]
pub extern "C" fn vrt_i_float(a: i64) -> f64 {
    a as f64
}
#[no_mangle]
pub extern "C" fn vrt_i_min(a: i64, b: i64) -> i64 {
    a.min(b)
}
#[no_mangle]
pub extern "C" fn vrt_i_max(a: i64, b: i64) -> i64 {
    a.max(b)
}
#[no_mangle]
pub extern "C" fn vrt_i_bnot(a: i64) -> i64 {
    !a
}
macro_rules! int_bits {
    ($($f:ident $name:literal),*) => {
        $(
            #[no_mangle]
            pub extern "C" fn $f(a: i64, b: i64) -> i64 {
                or_fail(ints::bits($name, a, b))
            }
        )*
    };
}
int_bits!(vrt_i_band "band", vrt_i_bor "bor", vrt_i_bxor "bxor", vrt_i_shl "shl", vrt_i_shr "shr", vrt_i_sar "sar",
    vrt_i_wrap_add "wrap_add", vrt_i_wrap_sub "wrap_sub", vrt_i_wrap_mul "wrap_mul", vrt_i_umulhi "umulhi");
// the little-endian readers: the interpreter's
// rules on the array's slots
macro_rules! le_readers {
    ($($f:ident $name:literal $t:ty),*) => {
        $(
            #[no_mangle]
            pub extern "C" fn $f(a: u32, off: i64) -> $t {
                match or_fail(super::bytes::read($name, ints_of(a), off)) {
                    super::bytes::Read::Int(i) => i as $t,
                    super::bytes::Read::Num(x) => x as $t,
                }
            }
        )*
    };
}
le_readers!(vrt_i_le_u8 "le_u8" i64, vrt_i_le_i8 "le_i8" i64, vrt_i_le_u16 "le_u16" i64, vrt_i_le_i16 "le_i16" i64,
    vrt_i_le_u32 "le_u32" i64, vrt_i_le_i32 "le_i32" i64, vrt_i_le_u64 "le_u64" i64, vrt_i_le_i64 "le_i64" i64,
    vrt_i_le_f16 "le_f16" f64, vrt_i_le_bf16 "le_bf16" f64, vrt_i_le_f32 "le_f32" f64, vrt_i_le_f64 "le_f64" f64);
/// `fround(x)`: the nearest binary32.
#[no_mangle]
pub extern "C" fn vrt_i_fround(x: f64) -> f64 {
    x as f32 as f64
}
/// `read_bytes(path, offset, count)`: a file the host attached, as an Int
/// array (width 1, 8 bytes a slot).
#[no_mangle]
pub extern "C" fn vrt_s_read_bytes(p: u32, off: f64, count: f64) -> u32 {
    let v = or_fail(super::bytes::read_bytes(wasm_str::str_of(p), off, count));
    let q = alloc(1, &[v.len()], v.len());
    ints_of(q).copy_from_slice(&v);
    q
}
#[no_mangle]
pub extern "C" fn vrt_i_ult(a: i64, b: i64) -> f64 {
    if ints::ult(a, b) { 1.0 } else { 0.0 }
}
#[no_mangle]
pub extern "C" fn vrt_i_str(a: i64, width: f64) -> u32 {
    wasm_str::new_str(&ints::text(a, width.clamp(0.0, 100.0) as usize))
}
#[no_mangle]
pub extern "C" fn vrt_i_fmt(a: i64, word: f64, width: f64) -> u32 {
    let f = crate::printfmt::Fmt::of_word(word as u32, 0, 0).unwrap_or(crate::printfmt::Fmt::REPR);
    let f = match (crate::printfmt::Fmt { flags: f.flags | crate::printfmt::FL_STAR_WIDTH, ..f }).with_stars(&[width]) {
        Ok(g) => crate::printfmt::Fmt { flags: f.flags, ..g },
        Err(m) => fail(m),
    };
    wasm_str::new_str(&crate::printfmt::int_text(a, f))
}
#[no_mangle]
pub extern "C" fn vrt_i_ustr(a: i64) -> u32 {
    wasm_str::new_str(&ints::ustr(a))
}
/// An Int array as `print` writes it: its slots read as Ints.
#[no_mangle]
pub extern "C" fn vrt_iarr_str(p: u32) -> u32 {
    let xs: Vec<i64> = data_of(p).iter().map(|x| x.to_bits() as i64).collect();
    wasm_str::new_str(&iarr_text(&xs))
}

// ---- enums ---------------------------------

/// The width an enum node's block is marked with: a block of `n` bytes, the
/// tag at 0, its place on the arena at 4, the payload's 8-byte slots from 8.
const EN_WIDTH: i32 = 5;

thread_local! {
    /// The number of the keep pass running: a node's block header holds the
    /// number of the pass that last reached it (at `EN_PASS`), so a node two
    /// values share is walked once, and the release keeps exactly the nodes
    /// marked with it. A release starts the next pass.
    static EN_PASS_NOW: std::cell::Cell<i32> = const { std::cell::Cell::new(1) };
}

/// Where an enum node's block header holds its pass (the rank's slot, which
/// a node's block does not use).
const EN_PASS: u32 = 8;

/// A new enum node of `bytes` bytes (8 for the tag and its place, 8 per
/// payload slot), filed on the arena; its data pointer.
#[no_mangle]
pub extern "C" fn vrt_en_new(bytes: u32) -> u32 {
    let pos = RT.with(|r| r.borrow().as_ref().expect("vrt_init runs first").arena.len());
    let p = alloc(EN_WIDTH, &[], bytes as usize) + HDR;
    unsafe { wr(p + 4, pos as i32) };
    p
}

/// A u32 of an enum descriptor (wasm_backend's `enum_descriptors`).
fn desc_word(at: u32) -> u32 {
    unsafe { std::ptr::read_unaligned(at as usize as *const u32) }
}

/// Variant `tag` of the enum whose descriptor is at `desc`: its name's offset
/// and length in the table, and each payload field's kind (with the
/// descriptor of an enum field).
fn en_variant(desc: u32, tag: u32) -> (u32, u32, Vec<(u32, u32)>) {
    let mut at = desc + 4;
    let nv = desc_word(desc);
    let mut found = (0, 0, Vec::new());
    for v in 0..nv {
        let (name, len, nf) = (desc_word(at), desc_word(at + 4), desc_word(at + 8));
        at += 12;
        let mut fields = Vec::new();
        for _ in 0..nf {
            let kind = desc_word(at);
            at += 4;
            let sub = if kind == 4 {
                at += 4;
                desc_word(at - 4)
            } else {
                0
            };
            fields.push((kind, sub));
        }
        if v == tag {
            found = (name, len, fields);
        }
    }
    found
}

/// An enum value as text, as the interpreter's `enum_text` writes it, with a
/// stack of what is left to write instead of recursion.
fn en_text(base: u32, desc: u32, p: u32, out: &mut String) {
    enum Job {
        Text(&'static str),
        Node(u32, u32),
        Field(u32, u32, u32),
    }
    let mut jobs = vec![Job::Node(desc, p)];
    while let Some(j) = jobs.pop() {
        match j {
            Job::Text(t) => out.push_str(t),
            Job::Node(desc, p) => {
                let tag = unsafe { rd(p) } as u32;
                let (name, len, fields) = en_variant(base + desc, tag);
                out.push_str(&string(base + name, len));
                if fields.is_empty() {
                    continue;
                }
                out.push('(');
                jobs.push(Job::Text(")"));
                for (k, (kind, sub)) in fields.iter().enumerate().rev() {
                    jobs.push(Job::Field(*kind, *sub, p + 8 + 8 * k as u32));
                    if k > 0 {
                        jobs.push(Job::Text(", "));
                    }
                }
            }
            Job::Field(kind, sub, slot) => unsafe {
                match kind {
                    0 => out.push_str(&format_g(std::ptr::read_unaligned(slot as usize as *const f64))),
                    1 => out.push_str(&std::ptr::read_unaligned(slot as usize as *const i64).to_string()),
                    2 => out.push_str(if std::ptr::read_unaligned(slot as usize as *const f64) != 0.0 { "true" } else { "false" }),
                    3 => out.push_str(&super::strings::repr(wasm_str::str_of(rd(slot) as u32))),
                    _ => jobs.push(Job::Node(sub, rd(slot) as u32)),
                }
            },
        }
    }
}

#[no_mangle]
pub extern "C" fn vrt_en_str(base: u32, desc: u32, p: u32) -> u32 {
    let mut t = String::new();
    en_text(base, desc, p, &mut t);
    wasm_str::new_str(&t)
}

thread_local! {
    /// Each descriptor read once: per variant, its payload fields' kinds
    /// (and an enum field's descriptor). Keyed by the descriptor's address,
    /// which a run's string table fixes.
    static EN_DESCS: RefCell<HashMap<u32, Rc<Vec<Vec<(u32, u32)>>>>> = RefCell::new(HashMap::new());
}

/// The payload kinds of every variant of the enum at `desc`, read once.
fn en_fields(desc: u32) -> Rc<Vec<Vec<(u32, u32)>>> {
    if let Some(d) = EN_DESCS.with(|m| m.borrow().get(&desc).cloned()) {
        return d;
    }
    let nv = desc_word(desc);
    let d: Rc<Vec<Vec<(u32, u32)>>> = Rc::new((0..nv).map(|v| en_variant(desc, v).2).collect());
    EN_DESCS.with(|m| m.borrow_mut().insert(desc, d.clone()));
    d
}

/// Keep an enum value: every node it reaches at or above the arena's `mark`
/// (one below it belongs to an older region), each once, and their strings.
#[no_mangle]
pub extern "C" fn vrt_keep_en(base: u32, desc: u32, p: u32, mark: u32) {
    let now = EN_PASS_NOW.with(|c| c.get());
    // the descriptors this walk has met (a list or a tree meets one or two)
    let mut met: Vec<(u32, Rc<Vec<Vec<(u32, u32)>>>)> = Vec::new();
    let mut todo = vec![(desc, p)];
    while let Some((desc, p)) = todo.pop() {
        if p == 0 || (unsafe { rd(p + 4) } as u32) < mark || unsafe { rd(p - HDR + EN_PASS) } == now {
            continue;
        }
        unsafe { wr(p - HDR + EN_PASS, now) };
        let tag = unsafe { rd(p) } as usize;
        let all = match met.iter().find(|(d, _)| *d == desc) {
            Some((_, a)) => a.clone(),
            None => {
                let a = en_fields(base + desc);
                met.push((desc, a.clone()));
                a
            }
        };
        for (k, (kind, sub)) in all[tag].iter().enumerate() {
            let slot = p + 8 + 8 * k as u32;
            match kind {
                3 => wasm_str::vrt_keep_str(unsafe { rd(slot) } as u32),
                4 => todo.push((*sub, unsafe { rd(slot) } as u32)),
                _ => {}
            }
        }
    }
}

// ---- fused trees and reductions over views (crate::fuse) -------------------
// The twin of the C runtime's viv_fz and viv_fz_red (fuse::C_RUNTIME): the
// module pushes the operands (`vrt_fz_a`, `vrt_fz_n`) in the order the
// unfused code evaluates them, then runs the program.

thread_local! {
    static FZ: RefCell<(Vec<u32>, Vec<f64>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}
const FZ_B: usize = 256;

#[no_mangle]
pub extern "C" fn vrt_fz_a(p: u32) {
    FZ.with(|f| f.borrow_mut().0.push(p));
}
#[no_mangle]
pub extern "C" fn vrt_fz_n(x: f64) {
    FZ.with(|f| f.borrow_mut().1.push(x));
}
/// The bounds of a slice read in place where it is used (`viv__slice_check`).
#[no_mangle]
pub extern "C" fn vrt_slicecheck(a: u32, spec: u32) -> f64 {
    if let Err(e) = super::slices::axes(&shape_of(a), data_of(spec)) {
        fail(e.msg);
    }
    0.0
}

/// One operation of a program: (op, how many numbers it spans).
fn fz_op(code: &[f64], i: usize) -> (u8, usize) {
    let op = code[i] as u8;
    (op, match op {
        0 | 1 | 3..=6 | 14 => 2,
        2 => 5,
        _ => 1,
    })
}

/// The operands the module pushed for this program, taken off the stacks.
fn fz_take(code: &[f64]) -> (Vec<u32>, Vec<f64>) {
    let (mut na, mut nn, mut i) = (0, 0, 0);
    while i < code.len() {
        let (op, w) = fz_op(code, i);
        match op {
            0 => na += 1,
            1 => nn += 1,
            2 => {
                na += 1;
                nn += 4 * code[i + 2] as usize;
            }
            _ => {}
        }
        i += w;
    }
    FZ.with(|f| {
        let mut f = f.borrow_mut();
        let (a0, n0) = (f.0.len() - na, f.1.len() - nn);
        (f.0.split_off(a0), f.1.split_off(n0))
    })
}

/// The walk in the unfused order (fuse::C_RUNTIME, viv_fz_walk): the shapes
/// left, and each operand array as (data address, first element, step); None
/// to run the unfused operations.
#[allow(clippy::type_complexity)]
fn fz_walk(code: &[f64], arrs: &[u32], nums: &[f64]) -> Option<(Vec<Vec<usize>>, Vec<(u32, i64, i64)>)> {
    let mut sh: Vec<Option<Vec<usize>>> = Vec::new();
    let mut lv = Vec::new();
    let mut i = 0;
    while i < code.len() {
        let (op, w) = fz_op(code, i);
        match op {
            0 => {
                let a = arrs[code[i + 1] as usize];
                lv.push((a, 0, 1));
                sh.push(Some(shape_of(a).to_vec()));
            }
            1 => sh.push(None),
            2 => {
                let a = arrs[code[i + 1] as usize];
                let (m, j) = (code[i + 2] as usize, code[i + 3] as usize);
                let ax = super::slices::axes(&shape_of(a), &nums[j..j + 4 * m]).unwrap_or_else(|e| fail(e.msg));
                let (start, st) = ax.affine()?;
                lv.push((a, start, st));
                sh.push(Some(ax.shape));
            }
            3..=6 => {
                let b = sh.pop().unwrap();
                if sh.last().unwrap() != &b {
                    return None;
                }
            }
            7..=13 => {
                sh.pop();
            }
            _ => {}
        }
        i += w;
    }
    Some((sh.into_iter().map(|s| s.unwrap_or_default()).collect(), lv))
}

/// Elements [b0, b0 + n) of the values the program leaves: slot k of `buf`
/// (FZ_B numbers each) holds stack entry k.
fn fz_block(code: &[f64], nums: &[f64], lv: &[(u32, i64, i64)], b0: usize, n: usize, buf: &mut [f64]) {
    let mut v = [0.0f64; 64];
    let (mut top, mut nl, mut i) = (0usize, 0usize, 0usize);
    while i < code.len() {
        let (op, w) = fz_op(code, i);
        match op {
            0 | 2 => {
                let (a, start, st) = lv[nl];
                nl += 1;
                let d = data_of(a);
                let o = &mut buf[top * FZ_B..top * FZ_B + n];
                for (k, x) in o.iter_mut().enumerate() {
                    *x = d[(start + (b0 + k) as i64 * st) as usize];
                }
                top += 1;
            }
            1 => {
                v[top] = nums[code[i + 1] as usize];
                top += 1;
            }
            3..=6 => {
                let (lo, hi) = buf.split_at_mut((top - 1) * FZ_B);
                let (a, b) = (&mut lo[(top - 2) * FZ_B..(top - 2) * FZ_B + n], &hi[..n]);
                let f = real_op(op as u32 - 3);
                for k in 0..n {
                    a[k] = f(a[k], b[k]);
                }
                top -= 1;
            }
            7..=13 => {
                let s = v[top - 1];
                let a = &mut buf[(top - 2) * FZ_B..(top - 2) * FZ_B + n];
                for x in a.iter_mut() {
                    *x = match op {
                        7 => *x + s,
                        8 => *x - s,
                        9 => s - *x,
                        10 => *x * s,
                        11 => *x / s,
                        12 => s / *x,
                        _ => x.vpow(s),
                    };
                }
                top -= 1;
            }
            14 => {
                let f = MAP_FNS[code[i + 1] as usize];
                for x in buf[(top - 1) * FZ_B..(top - 1) * FZ_B + n].iter_mut() {
                    *x = f(*x);
                }
            }
            _ => {}
        }
        i += w;
    }
}

/// The unfused operations, one at a time, in the unfused order.
fn fz_mat(code: &[f64], arrs: &[u32], nums: &[f64]) -> Vec<u32> {
    let mut a: Vec<u32> = Vec::new();
    let mut v: Vec<f64> = Vec::new();
    let mut i = 0;
    while i < code.len() {
        let (op, w) = fz_op(code, i);
        match op {
            0 => a.push(arrs[code[i + 1] as usize]),
            1 => v.push(nums[code[i + 1] as usize]),
            2 => {
                let (m, j) = (code[i + 2] as usize, code[i + 3] as usize);
                let spec = new_like(1, &[4 * m]);
                data_of(spec).copy_from_slice(&nums[j..j + 4 * m]);
                a.push(vrt_getslice(arrs[code[i + 1] as usize], spec));
            }
            3..=6 => {
                let b = a.pop().unwrap();
                let x = a.pop().unwrap();
                a.push(vrt_arith(op as u32 - 3, x, b));
            }
            7..=12 => {
                let s = v.pop().unwrap();
                let x = a.pop().unwrap();
                let (tag, first) = match op {
                    7 => (0, 0),
                    8 => (1, 0),
                    9 => (1, 1),
                    10 => (2, 0),
                    11 => (3, 0),
                    _ => (3, 1),
                };
                a.push(vrt_arith_s(tag, x, s, first));
            }
            13 => {
                let s = v.pop().unwrap();
                let x = a.pop().unwrap();
                a.push(vrt_pows(x, s));
            }
            14 => {
                let x = a.pop().unwrap();
                a.push(vrt_map(code[i + 1] as u32, x));
            }
            _ => {}
        }
        i += w;
    }
    a
}

/// A fused elementwise tree: the program is the array `code`.
#[no_mangle]
pub extern "C" fn vrt_fz(code: u32) -> u32 {
    let code = data_of(code).to_vec();
    let (arrs, nums) = fz_take(&code);
    let Some((sh, lv)) = fz_walk(&code, &arrs, &nums).filter(|(sh, _)| sh.len() == 1) else {
        return fz_mat(&code, &arrs, &nums)[0];
    };
    let r = new_like(1, &sh[0]);
    let size = data_of(r).len();
    let mut buf = vec![0.0; FZ_B * 64];
    let mut b0 = 0;
    while b0 < size {
        let n = (size - b0).min(FZ_B);
        fz_block(&code, &nums, &lv, b0, n, &mut buf);
        data_of(r)[b0..b0 + n].copy_from_slice(&buf[..n]);
        b0 += n;
    }
    r
}

/// A reduction (fuse::REDUCTIONS' numbers) over a program's one value, or
/// `dot` of its two, each in the order of the builtin it stands for.
#[no_mangle]
pub extern "C" fn vrt_fz_red(kind: u32, code: u32) -> f64 {
    let code = data_of(code).to_vec();
    let (arrs, nums) = fz_take(&code);
    let want = if kind == 5 { 2 } else { 1 };
    let walked = fz_walk(&code, &arrs, &nums).filter(|(sh, _)| sh.len() == want && (want == 1 || (sh[0].len() == 1 && sh[1].len() == 1 && sh[0] == sh[1])));
    let Some((sh, lv)) = walked else {
        let r = fz_mat(&code, &arrs, &nums);
        let d = data_of(r[0]);
        return match kind {
            0 => vrt_sum(r[0]),
            1 => d.iter().fold(0.0, |t, &x| t + x),
            2 => d.iter().fold(1.0, |t, &x| t * x),
            3 => super::arrays::npsum(d) / d.len() as f64,
            4 => d.iter().fold(0.0, |t, &x| t + x * x).sqrt(),
            5 => {
                let (su, sv) = (shape_of(r[0]).to_vec(), shape_of(r[1]).to_vec());
                if su.len() != 1 || sv.len() != 1 || su != sv {
                    fail(crate::errkind::kinded(crate::errkind::ErrKind::Value, format!("`dot` takes two vectors of one length (for matrices write A @ B); these have shapes {} and {}", super::arrays::shape_text(&su), super::arrays::shape_text(&sv))));
                }
                let e = data_of(r[1]);
                (0..d.len()).fold(0.0, |t, k| t + d[k] * e[k])
            }
            _ => vrt_extreme((kind == 6) as u32, r[0]),
        };
    };
    let size: usize = sh[0].iter().product();
    let mut buf = vec![0.0; FZ_B * 64];
    let mut t = match kind {
        2 => 1.0,
        6 => f64::NEG_INFINITY,
        7 => f64::INFINITY,
        _ => 0.0,
    };
    if kind == 0 || kind == 3 {
        // NumPy's order: blocks of 8192, each by the pairwise sum
        let mut chunk = vec![0.0; 8192];
        let mut c0 = 0;
        while c0 < size {
            let len = (size - c0).min(8192);
            let mut b = 0;
            while b < len {
                let n = (len - b).min(FZ_B);
                fz_block(&code, &nums, &lv, c0 + b, n, &mut buf);
                chunk[b..b + n].copy_from_slice(&buf[..n]);
                b += n;
            }
            t += super::arrays::npsum(&chunk[..len]);
            c0 += len;
        }
        return if kind == 3 { t / size as f64 } else { t };
    }
    let mut b0 = 0;
    while b0 < size {
        let n = (size - b0).min(FZ_B);
        fz_block(&code, &nums, &lv, b0, n, &mut buf);
        let x = &buf[..n];
        match kind {
            1 => x.iter().for_each(|&e| t += e),
            2 => x.iter().for_each(|&e| t *= e),
            4 => x.iter().for_each(|&e| t += e * e),
            5 => {
                let y = &buf[FZ_B..FZ_B + n];
                (0..n).for_each(|k| t += x[k] * y[k]);
            }
            6 => x.iter().for_each(|&e| t = crate::vmath::fmax(t, e)),
            _ => x.iter().for_each(|&e| t = crate::vmath::fmin(t, e)),
        }
        b0 += n;
    }
    if kind == 4 {
        t.sqrt()
    } else {
        t
    }
}

// ---- Float32 and the packed arrays ----------------------------
//
// The arithmetic is an f32 instruction in the module; these are the rest,
// each interp/f32s.rs's own code. A packed array is a raw byte block (width
// 0): its element count at +4 is its byte count, and dims[0] its length.

fn store_of(k: f64) -> crate::ast::Store {
    crate::ast::Store::of_code(k as u8)
}
fn packed_bytes<'a>(p: u32) -> &'a mut [u8] {
    unsafe { std::slice::from_raw_parts_mut((p + HDR) as usize as *mut u8, elems(p)) }
}
fn new_packed(k: crate::ast::Store, b: &[u8]) -> u32 {
    let p = alloc(0, &[b.len() / k.width()], b.len());
    packed_bytes(p).copy_from_slice(b);
    p
}
#[no_mangle]
pub extern "C" fn vrt_f_min(a: f32, b: f32) -> f32 {
    super::f32s::min(a, b)
}
#[no_mangle]
pub extern "C" fn vrt_f_max(a: f32, b: f32) -> f32 {
    super::f32s::max(a, b)
}
macro_rules! f32_funcs {
    ($($f:ident $name:literal),*) => {
        $(
            #[no_mangle]
            pub extern "C" fn $f(x: f32) -> f32 {
                super::f32s::func($name, x).unwrap_or(f32::NAN)
            }
        )*
    };
}
f32_funcs!(vrt_f_exp "exp", vrt_f_log "log", vrt_f_sin "sin", vrt_f_cos "cos", vrt_f_tanh "tanh");
#[no_mangle]
pub extern "C" fn vrt_f_bf16(x: f32) -> f32 {
    super::f32s::round_to(crate::ast::Store::BF16, x)
}
#[no_mangle]
pub extern "C" fn vrt_f_f16(x: f32) -> f32 {
    super::f32s::round_to(crate::ast::Store::F16, x)
}
#[no_mangle]
pub extern "C" fn vrt_f_repr(x: f32) -> u32 {
    wasm_str::new_str(&super::f32s::repr(x))
}
#[no_mangle]
pub extern "C" fn vrt_f_bare(x: f32) -> u32 {
    wasm_str::new_str(&crate::printfmt::bare(x as f64))
}
#[no_mangle]
pub extern "C" fn vrt_p_zeros(n: f64, k: f64) -> u32 {
    let k = store_of(k);
    let n = super::zeros_shape(&[n]).0.first().copied().unwrap_or(0);
    new_packed(k, &vec![0u8; n * k.width()])
}
#[no_mangle]
pub extern "C" fn vrt_p_getf(a: u32, i: f64, k: f64) -> f32 {
    let k = store_of(k);
    let b = packed_bytes(a);
    let at = or_fail(super::f32s::index(i, super::f32s::count(k, b)));
    match super::f32s::get(k, b, at) {
        super::f32s::Elt::F(x) => x,
        super::f32s::Elt::I(n) => n as f32,
    }
}
#[no_mangle]
pub extern "C" fn vrt_p_geti(a: u32, i: f64, k: f64) -> i64 {
    let k = store_of(k);
    let b = packed_bytes(a);
    let at = or_fail(super::f32s::index(i, super::f32s::count(k, b)));
    match super::f32s::get(k, b, at) {
        super::f32s::Elt::I(n) => n,
        super::f32s::Elt::F(x) => x as i64,
    }
}
#[no_mangle]
pub extern "C" fn vrt_p_setf(a: u32, i: f64, v: f32, k: f64) -> f64 {
    let k = store_of(k);
    let b = packed_bytes(a);
    let at = or_fail(super::f32s::index(i, super::f32s::count(k, b)));
    or_fail(super::f32s::set(k, b, at, super::f32s::Elt::F(v)));
    0.0
}
#[no_mangle]
pub extern "C" fn vrt_p_seti(a: u32, i: f64, v: i64, k: f64) -> f64 {
    let k = store_of(k);
    let b = packed_bytes(a);
    let at = or_fail(super::f32s::index(i, super::f32s::count(k, b)));
    or_fail(super::f32s::set(k, b, at, super::f32s::Elt::I(v)));
    0.0
}
#[no_mangle]
pub extern "C" fn vrt_p_len(a: u32) -> f64 {
    shape_of(a).dims[0] as f64
}
#[no_mangle]
pub extern "C" fn vrt_p_conv(a: u32, from: f64, to: f64) -> u32 {
    let (from, to) = (store_of(from), store_of(to));
    let b = or_fail(super::f32s::convert(from, to, packed_bytes(a)));
    new_packed(to, &b)
}
#[no_mangle]
pub extern "C" fn vrt_p_of_arr(a: u32, k: f64) -> u32 {
    let k = store_of(k);
    let b = super::f32s::pack_floats(k, data_of(a).iter().map(|x| *x as f32));
    new_packed(k, &b)
}
#[no_mangle]
pub extern "C" fn vrt_p_of_iarr(a: u32, k: f64) -> u32 {
    let k = store_of(k);
    let b = or_fail(super::f32s::pack_ints(k, ints_of(a).iter().copied()));
    new_packed(k, &b)
}
#[no_mangle]
pub extern "C" fn vrt_p_to_arr(a: u32, k: f64) -> u32 {
    let k = store_of(k);
    let xs = super::f32s::floats(k, packed_bytes(a));
    let p = alloc(1, &[xs.len()], xs.len());
    for (d, x) in data_of(p).iter_mut().zip(xs) {
        *d = x as f64;
    }
    p
}
#[no_mangle]
pub extern "C" fn vrt_p_to_iarr(a: u32, k: f64) -> u32 {
    let k = store_of(k);
    let b = packed_bytes(a);
    let n = super::f32s::count(k, b);
    let p = alloc(1, &[n], n);
    for (i, d) in ints_of(p).iter_mut().enumerate() {
        if let super::f32s::Elt::I(v) = super::f32s::get(k, b, i) {
            *d = v;
        }
    }
    p
}
#[no_mangle]
pub extern "C" fn vrt_p_sum(a: u32, k: f64) -> f32 {
    let k = store_of(k);
    super::f32s::sum(&super::f32s::floats(k, packed_bytes(a)))
}
#[no_mangle]
pub extern "C" fn vrt_p_isum(a: u32, k: f64) -> i64 {
    let k = store_of(k);
    or_fail(super::f32s::isum(k, packed_bytes(a)))
}
#[no_mangle]
pub extern "C" fn vrt_p_dot(a: u32, b: u32) -> f32 {
    let k = crate::ast::Store::F32;
    or_fail(super::f32s::dot(&super::f32s::floats(k, packed_bytes(a)), &super::f32s::floats(k, packed_bytes(b))))
}
#[no_mangle]
pub extern "C" fn vrt_p_str(a: u32, k: f64) -> u32 {
    let k = store_of(k);
    wasm_str::new_str(&super::f32s::text(k, packed_bytes(a)))
}
