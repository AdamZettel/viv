#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
#
# Build Viv from this checkout and run a first program, an atlas and a call
# from Python. Run it again at any time: it installs nothing, needs no sudo,
# and writes only under target/ (and vipyr's cache, examples/__pycache__/).
#
#     sh quickstart.sh
#
# Needs Rust 1.77 or newer (cargo and rustc), a C compiler, Python 3.9 or
# newer and NumPy. A missing one is reported with how to install it.
set -eu

root=$(cd "$(dirname "$0")" && pwd)
cd "$root"
out=target/quickstart
viv=target/release/viv

step() { printf '\n== %s\n' "$*"; }
show() { printf '$ %s\n' "$*"; }

# ---------------------------------------------------------------- prerequisites
step "Checking prerequisites"
missing=0
os=$(uname -s)

if command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
    rv=$(rustc --version 2>/dev/null | awk '{print $2}')
    major=$(echo "${rv:-0.0}" | cut -d. -f1)
    minor=$(echo "${rv:-0.0}" | cut -d. -f2)
    if [ "$major" -gt 1 ] || { [ "$major" -eq 1 ] && [ "$minor" -ge 77 ]; }; then
        echo "ok   Rust $rv"
    else
        echo "old  Rust $rv: Viv needs 1.77 or newer. Update with: rustup update stable"
        missing=1
    fi
else
    echo "no   Rust (cargo and rustc). Install it from https://rustup.rs:"
    echo "       curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    missing=1
fi

cc_found=""
for c in "${CC:-}" cc clang gcc; do
    if [ -n "$c" ] && command -v "$c" >/dev/null 2>&1; then cc_found=$c; break; fi
done
if [ -n "$cc_found" ]; then
    echo "ok   C compiler ($cc_found)"
else
    echo "no   C compiler (cc, clang or gcc). Install one:"
    if [ "$os" = Darwin ]; then
        echo "       xcode-select --install"
    else
        echo "       Debian or Ubuntu: sudo apt install build-essential"
        echo "       Fedora: sudo dnf install gcc"
    fi
    missing=1
fi

py=""
for p in python3 python; do
    if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then
        py=$p; break
    fi
done
if [ -n "$py" ]; then
    echo "ok   Python $("$py" -c 'import platform; print(platform.python_version())')"
    if "$py" -c 'import numpy' 2>/dev/null; then
        echo "ok   NumPy $("$py" -c 'import numpy; print(numpy.__version__)')"
    else
        echo "no   NumPy. Install it with: $py -m pip install numpy"
        missing=1
    fi
else
    echo "no   Python 3.9 or newer. Install it:"
    if [ "$os" = Darwin ]; then
        echo "       brew install python    (or the installer from https://www.python.org)"
    else
        echo "       Debian or Ubuntu: sudo apt install python3 python3-pip"
        echo "       Fedora: sudo dnf install python3 python3-pip"
    fi
    missing=1
fi

if [ "$missing" -ne 0 ]; then
    echo
    echo "Install what is missing above, then run sh quickstart.sh again."
    exit 1
fi

# ---------------------------------------------------------------- build
step "Building the compiler (the first build fetches crates and takes a few minutes)"
mkdir -p "$out"
show cargo build --release -p viv
if cargo build --release -p viv >"$out/build.log" 2>&1; then
    echo "built $viv (the compiler's messages are in $out/build.log)"
else
    tail -n 40 "$out/build.log"
    echo
    echo "The build failed; the whole log is $out/build.log."
    exit 1
fi

# ---------------------------------------------------------------- first program
step "A first program, in the interpreter"
show "$viv" run -q examples/hello.viv
"$viv" run -q examples/hello.viv

step "The same program as C"
show "$viv" examples/hello.viv -o "$out/hello.c"
"$viv" examples/hello.viv -o "$out/hello.c"
show "$cc_found" -O2 "$out/hello.c" -lm -o "$out/hello"
"$cc_found" -O2 "$out/hello.c" -lm -o "$out/hello"
show "$out/hello"
"$out/hello"

# ---------------------------------------------------------------- atlas
step "An atlas of MINPACK's Levenberg-Marquardt solver on Rosenbrock's problem"
show "$viv" atlas lib/minpack/story_rosenbrock.viv -o "$out/rosenbrock.atlas.html"
"$viv" atlas lib/minpack/story_rosenbrock.viv -o "$out/rosenbrock.atlas.html"

# ---------------------------------------------------------------- vipyr
step "The same def, called from Python through vipyr"
cat <<'EOF'
$ PYTHONPATH=python python3
import math
import vipyr as vp
hello = vp.load("examples/hello.viv")
print(hello.newton_sqrt(2.0), math.sqrt(2.0))
print(hello.newton_sqrt(3.0), math.sqrt(3.0))
EOF
VIV="$root/$viv" PYTHONPATH=python "$py" - <<'EOF'
import math
import vipyr as vp
hello = vp.load("examples/hello.viv")
print(hello.newton_sqrt(2.0), math.sqrt(2.0))
print(hello.newton_sqrt(3.0), math.sqrt(3.0))
EOF

# ---------------------------------------------------------------- next
step "Next"
cat <<EOF
Open the atlas in a browser: $out/rosenbrock.atlas.html
Put the compiler on your PATH: cargo install --path lang/viv
Commands: $viv --help
A ten-minute tour from the command line: docs/tutorial.md
vipyr, Viv from Python: python/vipyr/README.md
The site, with the workbench, the docs and the tour: "The site" in README.md
EOF
