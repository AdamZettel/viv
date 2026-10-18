// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! I/O is injected, never assumed (1.8). The compiler core reads sources,
//! writes outputs, and reaches external toolchains only through an
//! [`IoEnvironment`]. The CLI supplies a native one; tests supply an in-memory
//! one; a future WASM build supplies one that bridges to JS — and none of the
//! core changes.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The injected boundary between the compiler core and the outside world.
pub trait IoEnvironment {
    fn read_to_string(&self, path: &Path) -> std::io::Result<String>;
    fn write(&self, path: &Path, contents: &[u8]) -> std::io::Result<()>;

    /// An external LLVM-based toolchain (clang), if one is available. Absent in
    /// the browser — which is exactly why the direct WASM backend exists. The
    /// two-tier backend story keys off this `Option`.
    fn toolchain(&self) -> Option<&dyn Toolchain> {
        None
    }
}

/// The target a [`Toolchain`] compiles kit-emitted C to.
#[derive(Clone, Debug)]
pub enum Target {
    Wasm32,
    /// A native target triple (e.g. `"aarch64-apple-darwin"`).
    Native(String),
}

#[derive(Clone, Copy, Debug)]
pub enum OptLevel {
    O0,
    O2,
    O3,
}

/// Compiles kit-emitted C to a target object/module. Injected, never assumed
/// (1.8): the kit core never spawns clang itself.
pub trait Toolchain {
    fn compile_c(&self, c: &str, target: Target, opt: OptLevel) -> Result<Vec<u8>, String>;
}

/// The native filesystem. Used by the CLI binary.
pub struct NativeIo;

impl IoEnvironment for NativeIo {
    fn read_to_string(&self, path: &Path) -> std::io::Result<String> {
        std::fs::read_to_string(path)
    }
    fn write(&self, path: &Path, contents: &[u8]) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, contents)
    }
}

/// An in-memory environment for tests: reads come from a preloaded map, writes
/// are captured. Demonstrates that the core is platform-agnostic by construction.
#[derive(Default)]
pub struct MemoryIo {
    pub inputs: HashMap<PathBuf, String>,
    pub outputs: std::cell::RefCell<HashMap<PathBuf, Vec<u8>>>,
}

impl MemoryIo {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_input(mut self, path: impl Into<PathBuf>, text: impl Into<String>) -> Self {
        self.inputs.insert(path.into(), text.into());
        self
    }
}

impl IoEnvironment for MemoryIo {
    fn read_to_string(&self, path: &Path) -> std::io::Result<String> {
        self.inputs.get(path).cloned().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, format!("no such input: {}", path.display()))
        })
    }
    fn write(&self, path: &Path, contents: &[u8]) -> std::io::Result<()> {
        self.outputs.borrow_mut().insert(path.to_path_buf(), contents.to_vec());
        Ok(())
    }
}
