#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# The local gate runner: the quick tier on each new main commit, the nightly
# tier once a night after 03:00, both on a clean checkout of main in their own
# worktree (../viv-wt/.gate-main, kept so its target/ stays built). Results go
# to target/gate/history.jsonl of the main repository like any gate run;
# `tools/gate.sh status` shows them.
#
#     tools/gate/runner.sh tick      what launchd runs every 10 minutes: a nightly
#                                    if one is due, else a quick if main moved
#     tools/gate/runner.sh quick     the quick tier on main now
#     tools/gate/runner.sh nightly   the nightly tier on main now (waits up to
#                                    2 hours for a timing lock to clear)
#     tools/gate/runner.sh plist     write the launchd agent and print how to install it
set -eu
here=$(cd "$(dirname "$0")/../.." && pwd)
common=$(cd "$here" && cd "$(git rev-parse --git-common-dir)/.." && pwd)
state="$common/target/gate"
wt="$(dirname "$common")/viv-wt/.gate-main"
mkdir -p "$state"

checkout() {
    sha=$(git -C "$common" rev-parse main)
    if [ -d "$wt" ] && git -C "$wt" rev-parse --git-dir > /dev/null 2>&1; then
        git -C "$wt" checkout -q --detach -f "$sha"
        git -C "$wt" clean -q -fd
    else
        git -C "$common" worktree add -q --detach "$wt" "$sha"
    fi
}

locked() {
    git -C "$common" worktree list --porcelain | sed -n 's/^worktree //p' | while read -r p; do
        [ -e "$p/study/ports/.timing.lock" ] && echo "$p"
    done | grep -q .
}

run() {   # run <tier>
    tier=$1
    lock="$state/runner.lock"
    mkdir "$lock" 2> /dev/null || { echo "runner: a run is in progress ($lock)"; return 0; }
    trap 'rmdir "$lock" 2> /dev/null || true' EXIT
    if [ "$tier" = nightly ]; then
        waited=0
        while locked && [ $waited -lt 7200 ]; do sleep 300; waited=$((waited + 300)); done
    fi
    checkout
    echo "runner: $tier on main $(git -C "$wt" rev-parse --short HEAD) at $(date '+%F %T')"
    GATE_SOURCE=runner "$wt/tools/gate.sh" "$tier" --keep-going > "$state/runner-$tier.log" 2>&1 || true
    tail -4 "$state/runner-$tier.log"
    git -C "$wt" rev-parse HEAD > "$state/runner-$tier.sha"
    [ "$tier" = nightly ] && date +%F > "$state/runner-nightly.date"
    rmdir "$lock" 2> /dev/null || true
}

case "${1:-tick}" in
    tick)
        today=$(date +%F); hour=$(date +%H)
        last_night=$(cat "$state/runner-nightly.date" 2> /dev/null || echo none)
        if [ "$hour" -ge 3 ] && [ "$hour" -lt 9 ] && [ "$last_night" != "$today" ]; then
            run nightly
        elif [ "$(git -C "$common" rev-parse main)" != "$(cat "$state/runner-quick.sha" 2> /dev/null || echo none)" ]; then
            run quick
        fi ;;
    quick|nightly) run "$1" ;;
    plist)
        out="$state/org.viv-lang.gate.plist"
        path="$HOME/.cargo/bin:$(dirname "$(command -v python3)"):$(dirname "$(command -v node)"):/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        sed -e "s#@RUNNER@#$common/tools/gate/runner.sh#" -e "s#@PATH@#$path#" -e "s#@LOG@#$state/launchd.log#" \
            "$here/tools/gate/org.viv-lang.gate.plist.in" > "$out"
        echo "wrote $out (not installed). To install:"
        echo "  cp $out ~/Library/LaunchAgents/ && launchctl bootstrap gui/\$(id -u) ~/Library/LaunchAgents/org.viv-lang.gate.plist"
        echo "To remove:"
        echo "  launchctl bootout gui/\$(id -u)/org.viv-lang.gate && rm ~/Library/LaunchAgents/org.viv-lang.gate.plist" ;;
    *) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
