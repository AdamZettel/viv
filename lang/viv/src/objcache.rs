// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The object cache of `viv build`.
//!
//! A split emission (`viv f.viv --split <dir>`) writes the program's C as
//! translation units: one per imported module, whose text depends only on
//! that module's defs and what they use, and the program's own unit. Each
//! unit is compiled to an object once per distinct triple of unit text,
//! compiler flags and compiler identity, and kept under that triple's SHA-256 in the cache directory; a later build
//! that produces the same text with the same flags and the same compiler
//! links the kept object. The key is the whole text, so whatever changes the
//! C of a unit (the module, a callee's signature, a fact the compiler proved
//! from a call site, the runtime) changes the key: a stale object cannot be
//! reused. Each mode is its own entry, since its flags differ.
//!
//! The cache directory is `$VIV_CACHE_DIR`, else `$XDG_CACHE_HOME/viv`, else
//! `~/.cache/viv`; objects go in `obj/<first two hex digits>/<key>.o`.
//! Nothing is evicted: `viv build --cache-clear` empties it.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The SHA-256 of `data` as 64 hex digits (FIPS 180-4).
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
        0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
        0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut msg = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bits.to_be_bytes());
    for block in msg.chunks(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

/// Where the objects are kept.
pub fn cache_dir() -> PathBuf {
    if let Ok(d) = std::env::var("VIV_CACHE_DIR") {
        if !d.is_empty() {
            return PathBuf::from(d);
        }
    }
    if let Ok(d) = std::env::var("XDG_CACHE_HOME") {
        if !d.is_empty() {
            return PathBuf::from(d).join("viv");
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache").join("viv")
}

/// What identifies the C compiler: `CC`'s name and everything `--version`
/// prints (its version, its target and its install directory).
pub fn compiler_identity(cc: &str) -> String {
    let v = Command::new(cc).arg("--version").output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
    format!("{cc}\n{v}")
}

/// The cache key of a unit.
pub fn key(text: &str, flags: &[String], identity: &str) -> String {
    let mut data = Vec::with_capacity(text.len() + 256);
    data.extend_from_slice(b"viv-object-1\0");
    data.extend_from_slice(identity.as_bytes());
    data.push(0);
    data.extend_from_slice(flags.join(" ").as_bytes());
    data.push(0);
    data.extend_from_slice(text.as_bytes());
    sha256_hex(&data)
}

/// The object of `key` in `dir`.
pub fn object_path(dir: &Path, key: &str) -> PathBuf {
    dir.join("obj").join(&key[..2]).join(format!("{key}.o"))
}

/// What happened to one unit in a build.
pub struct Compiled {
    pub name: String,
    pub object: PathBuf,
    /// Taken from the cache (no compiler run).
    pub hit: bool,
    /// Seconds the compiler took (0 on a hit).
    pub secs: f64,
}

/// Compile `c` (a unit's C file) to an object with `flags`, through the
/// cache at `dir` unless `use_cache` is false (then into `scratch`). A new
/// object is written under a temporary name and renamed into place, so a
/// build running at the same time sees a whole object or none.
pub fn compile(cc: &str, identity: &str, flags: &[String], c: &Path, dir: &Path, use_cache: bool, scratch: &Path) -> Result<Compiled, String> {
    let name = c.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let text = std::fs::read_to_string(c).map_err(|e| format!("error: cannot read {}: {e}", c.display()))?;
    let k = key(&text, flags, identity);
    let obj = if use_cache { object_path(dir, &k) } else { scratch.join(format!("{name}.o")) };
    if use_cache && obj.is_file() {
        return Ok(Compiled { name, object: obj, hit: true, secs: 0.0 });
    }
    if let Some(p) = obj.parent() {
        std::fs::create_dir_all(p).map_err(|e| format!("error: cannot make {}: {e}", p.display()))?;
    }
    // one compiler per key at a time: builds started together (the gate
    // runs many) wait for the first instead of all compiling the same unit
    let _lock = if use_cache { Lock::take(&obj) } else { None };
    if use_cache && obj.is_file() {
        return Ok(Compiled { name, object: obj, hit: true, secs: 0.0 });
    }
    let tmp = obj.with_extension(format!("o.{}.{:?}.tmp", std::process::id(), std::thread::current().id()).replace(['(', ')'], ""));
    let t0 = std::time::Instant::now();
    let out = Command::new(cc).args(flags).arg("-c").arg(c).arg("-o").arg(&tmp).output().map_err(|e| format!("error: cannot run {cc}: {e}"))?;
    let secs = t0.elapsed().as_secs_f64();
    if !out.status.success() {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("error: {cc} rejected the generated C of {} :\n{}", c.display(), String::from_utf8_lossy(&out.stderr)));
    }
    std::fs::rename(&tmp, &obj).map_err(|e| format!("error: cannot keep {}: {e}", obj.display()))?;
    Ok(Compiled { name, object: obj, hit: false, secs })
}

/// A key's lock file, `<object>.lock`, created exclusively and removed when
/// dropped. Taking it waits while another build holds it; a lock older than
/// `LOCK_STALE` seconds is from a build that died and is taken over.
struct Lock(PathBuf);

const LOCK_STALE: u64 = 900;

impl Lock {
    fn take(obj: &Path) -> Option<Lock> {
        let path = obj.with_extension("o.lock");
        loop {
            match std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(_) => return Some(Lock(path)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if obj.is_file() {
                        return None;
                    }
                    let age = std::fs::metadata(&path).and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).map_or(0, |d| d.as_secs());
                    if age > LOCK_STALE {
                        let _ = std::fs::remove_file(&path);
                        continue;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(200));
                }
                // a cache we cannot lock in is still a cache: compile without it
                Err(_) => return None,
            }
        }
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Compile every C file in `files`, `jobs` at a time.
pub fn compile_all(cc: &str, flags: &[String], files: &[PathBuf], dir: &Path, use_cache: bool, scratch: &Path, jobs: usize) -> Result<Vec<Compiled>, String> {
    let identity = compiler_identity(cc);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: std::sync::Mutex<Vec<Option<Result<Compiled, String>>>> = std::sync::Mutex::new((0..files.len()).map(|_| None).collect());
    std::thread::scope(|s| {
        for _ in 0..jobs.max(1).min(files.len().max(1)) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= files.len() {
                    break;
                }
                let r = compile(cc, &identity, flags, &files[i], dir, use_cache, scratch);
                results.lock().unwrap()[i] = Some(r);
            });
        }
    });
    results.into_inner().unwrap().into_iter().map(|r| r.unwrap_or_else(|| Err("error: a unit was not compiled".to_string()))).collect()
}

/// How many compilers to run at once: `$VIV_JOBS`, else the machine's
/// parallelism, at most 8.
pub fn jobs() -> usize {
    if let Some(n) = std::env::var("VIV_JOBS").ok().and_then(|v| v.parse::<usize>().ok()) {
        return n.max(1);
    }
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_answers() {
        assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    /// Every mutable file-scope variable of the runtime is either shared by
    /// the units of a split build (defined in the program's unit, `extern`
    /// in the others) or a table each unit keeps for itself. A new one in
    /// neither list would be one copy per unit, silently: this fails first.
    #[test]
    fn split_state_covers_the_runtime() {
        use crate::lower::{static_data_line, PER_UNIT_STATE, SHARED_STATE};
        let text = crate::lower::runtime_c_text();
        let mut unlisted = Vec::new();
        for line in text.lines() {
            if let Some((names, mutable)) = static_data_line(line) {
                for n in names {
                    if mutable && !SHARED_STATE.contains(&n.as_str()) && !PER_UNIT_STATE.contains(&n.as_str()) {
                        unlisted.push(n);
                    }
                }
            } else if line.starts_with("static ") && !line.contains('(') && line.trim_end().ends_with('{') && !line.starts_with("static const ") {
                // a mutable table over several lines
                unlisted.push(line.to_string());
            }
        }
        assert!(unlisted.is_empty(), "mutable runtime state in neither SHARED_STATE nor PER_UNIT_STATE: {unlisted:?}");
        // and every listed name is still in the runtime
        for n in SHARED_STATE {
            assert!(text.contains(n), "{n} is listed in SHARED_STATE but no longer in the runtime");
        }
    }

    #[test]
    fn the_key_covers_text_flags_and_compiler() {
        let f = vec!["-O2".to_string()];
        let k = key("int x;", &f, "cc 1");
        assert_ne!(k, key("int y;", &f, "cc 1"));
        assert_ne!(k, key("int x;", &["-O3".to_string()], "cc 1"));
        assert_ne!(k, key("int x;", &f, "cc 2"));
        assert_eq!(k, key("int x;", &f, "cc 1"));
    }
}
