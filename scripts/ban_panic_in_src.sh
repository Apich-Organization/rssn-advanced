#!/usr/bin/env bash
# Fails CI if non-test code in `src/` uses panic-introducing constructs
# (`.unwrap()`, `.expect(...)`, `assert!`, `assert_eq!`, `assert_ne!`,
# `panic!`, `todo!`, `unreachable!`).
#
# Rationale: every audit in `review/` flagged this exact pattern as
# the source of unrecoverable runtime aborts in the symbolic compute
# core (`plan.md §4.3`). Phase 7 of the dev plan replaces these with
# `cold_*` error returns or Option/Result short-circuits.
#
# Allowed exceptions (still flagged but human-confirmed):
#  * `const _: () = assert!(...)` — compile-time, no runtime panic.
#  * Any line carrying an inline `// allow-panic: <reason>` annotation.
#  * Lines inside a `#[cfg(test)]` block — automatically skipped using
#    a per-file scan (we cut the file at the first `#[cfg(test)]` and
#    only check the prefix).
#
# Init-path failures (e.g. `JitCompiler::new()` Cranelift setup)
# legitimately abort because the rest of the system cannot proceed
# without an ISA. Those carry `// allow-panic: init-only` comments.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Patterns that introduce a runtime panic.
PANIC_RE='\.unwrap\(\)|\.expect\(|assert_eq!|assert_ne!|^[[:space:]]*assert!|^[[:space:]]*panic!|^[[:space:]]*todo!|^[[:space:]]*unreachable!|unreachable!\('

offenders=""
for f in $(find src/ -name "*.rs" | sort); do
    [ ! -f "$f" ] && continue

    # Determine the prefix range to scan: everything BEFORE the first
    # `#[cfg(test)]` line (test code is exempt).
    test_start=$(grep -n '^[[:space:]]*#\[cfg(test)\]' "$f" 2>/dev/null | head -1 | cut -d: -f1 || true)
    if [ -n "$test_start" ]; then
        end_line=$((test_start - 1))
    else
        end_line=$(wc -l < "$f")
    fi
    [ "$end_line" -le 0 ] && continue

    # Scan the prefix, filter exemptions.
    raw_matches=$(head -n "$end_line" "$f" \
        | grep -nE "$PANIC_RE" \
        | grep -v 'allow-panic:' \
        | grep -vE '^[0-9]+:[[:space:]]*//' \
        | grep -vE 'const[[:space:]]+_:[[:space:]]*\(\)[[:space:]]*=[[:space:]]*assert' \
        || true)

    # For each remaining offender, scan the 8 lines immediately above
    # the panic site for an `allow-panic:` annotation. 8 lines is
    # enough to cover multi-line statements where the annotation
    # comment sits above a `let foo = bar.baz().expect(…);` chain.
    matches=""
    while IFS= read -r m; do
        [ -z "$m" ] && continue
        lineno="${m%%:*}"
        start_line=$((lineno - 8))
        [ "$start_line" -lt 1 ] && start_line=1
        end_window=$((lineno - 1))
        if [ "$end_window" -ge "$start_line" ]; then
            window=$(sed -n "${start_line},${end_window}p" "$f")
            if echo "$window" | grep -q 'allow-panic:'; then
                continue
            fi
        fi
        matches+="$m"$'\n'
    done <<< "$raw_matches"

    # Strip trailing newline so empty `matches` are detected cleanly.
    matches="${matches%$'\n'}"

    if [ -n "$matches" ]; then
        offenders+="$f:"$'\n'"$matches"$'\n'
    fi
done

if [ -z "$offenders" ]; then
    echo "OK: no panic-inducing constructs in non-test code under src/."
    exit 0
fi

echo "ERROR: panic-inducing constructs detected in non-test code."
echo "Each offender must either propagate a Result/Option or carry an"
echo "inline '// allow-panic: <reason>' annotation."
echo
echo "$offenders"
exit 1
