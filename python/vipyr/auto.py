# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""`import vipyr.auto` installs the import hook: `import kernels` then
finds `kernels.viv` beside your .py files and inside packages."""
from ._hook import install as _install

_install()
