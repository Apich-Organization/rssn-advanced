#!/usr/bin/env bash
# Fails CI if any code path in `src/` uses `Ordering::SeqCst` (or its
# spelled-out variant) — `plan.md §4.2` forbids it.
#
# Acquire/Release is the only permitted ordering pair for the JIT and
# storage hot paths. SeqCst forces a full memory barrier on every
# operation, which kills throughput on weak-memory targets (aarch64,
# riscv64) and adds redundant fences on x86_64 too.
#
# Whitelist `// allow-seqcst: <reason>` on the same line for cases where
# SeqCst is genuinely needed (cross-architecture lock-free primitive that
# can't be expressed otherwise). The whitelist token must include a
# non-empty reason.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Find candidate lines. Match only real code use (`Ordering::SeqCst`)
# so doc-comment mentions of the word don't trip the gate.
candidates=$(grep -rn 'Ordering::SeqCst\|::SeqCst[^a-zA-Z_]' src/ tests/ benches/ 2>/dev/null \
    | grep -v 'allow-seqcst:' \
    | grep -v '^[^:]*:[0-9]*:\s*//' \
    || true)

if [[ -z "$candidates" ]]; then
    echo "OK: no SeqCst usage in src/, tests/, benches/."
    exit 0
fi

echo "ERROR: SeqCst usage detected (plan.md §4.2 forbids this)."
echo "Each offender must either move to Acquire/Release or carry"
echo "an inline '// allow-seqcst: <reason>' annotation."
echo
echo "$candidates"
exit 1
