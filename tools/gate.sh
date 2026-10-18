#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Viv's gate (tools/gate/gate.py has the details):
#
#     tools/gate.sh quick|premerge|postmerge|full|nightly [--list] [--json] [--keep-going]
#                   [--only a,b] [--ignore-lock] [--no-cache] [--sample 5%]
#                   [--base REV] [--no-triggers] [--serial]
#     tools/gate.sh status [N]      the last N runs, from target/gate/history.jsonl
#     tools/gate.sh cache [stats|prune [--days N]|clear]   the cache's store (tools/gate/cache.py)
#     tools/gate.sh bugs [--check]  the bug ledger's trend (bugs/ledger.jsonl)
#     tools/gate.sh ratchet [--update]
#     tools/gate.sh triggers        the post-merge suites this branch's paths add to premerge
#
# quick: the language checks, before every commit you report. premerge: what
# a merge needs (tools/merge.sh runs it on the merged tree): quick, the
# library engine-agreement set, a smoke program per application family, and
# the post-merge suites the branch's paths trigger. postmerge: the
# application suites, py2viv, memcheck-quick, www-check, hub-golden. full:
# premerge + postmerge. nightly: full plus the long and timing checks, cache
# off. Independent checks run at once within a core and memory budget.
# Exits nonzero when any check fails or is blocked.
#
# Every tier but nightly reuses a passing result whose inputs have not changed
# (tools/gate/cache.py), rerunning a 5% sample of what it reuses; nightly and
# --no-cache run everything.
exec python3 "$(dirname "$0")/gate/gate.py" "$@"
