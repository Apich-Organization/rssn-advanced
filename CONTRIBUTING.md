# Contributing to rssn-advanced

Thank you for your interest in contributing!  Every contribution — from a
one-line typo fix to a new optimisation pass — is genuinely appreciated.

This document covers how to get started, what the code-style requirements
are, how to run tests and benchmarks, and how the PR review process works.

---

## Table of contents

1. [Project overview for contributors](#1-project-overview-for-contributors)
2. [Getting started](#2-getting-started)
3. [Development workflow](#3-development-workflow)
4. [Code style requirements](#4-code-style-requirements)
5. [Testing](#5-testing)
6. [Benchmarking](#6-benchmarking)
7. [Pull request checklist](#7-pull-request-checklist)
8. [Good first issues](#8-good-first-issues)
9. [Communication](#9-communication)

---

## 1. Project overview for contributors

rssn-advanced is a **symbolic expression engine** with the following main
pipeline:

```
string expression
    → parser (nom)
    → DagBuilder (hash-consed DAG)
    → AstProjection (local tree, relative pointers)
    → JitCompiler (Cranelift) → native fn ptr
                              → batch fn ptr  (2-row ILP)
    → HeuristicEngine (rule-registry simplifier)
    → EGraph (equality saturation + cost extraction)
```

Custom operators plug into all three back-ends at once via a
`CustomOpDescriptor` stored in a `CustomOpRegistry`.

**Key design constraints:**

- **Zero `unwrap()` / `expect()` on the hot path.**  Error handling uses
  `#[cold] #[inline(never)]` constructors from the `error` module.
- **No intermediate heap allocations in the JIT-compiled expression.**
  The entire expression must live in CPU registers.
- **`extern "C"` ABI for all public evaluation functions** so the JIT can
  emit a direct `call` without a trampoline.
- **All public items must have doc comments** — the crate denies
  `missing_docs`.
- **The linter bar is intentionally high** — see §4.

---

## 2. Getting started

**Requirements:**

| Tool | Minimum version |
|------|----------------|
| Rust (stable) | 1.93.0 |
| Rust (nightly) | latest — for `cargo +nightly fmt` |
| Python 3 | 3.10+ — only needed to run the NumPy comparison benchmark |

**Steps:**

```bash
# 1. Fork and clone
git clone https://github.com/Apich-Organization/rssn-advanced.git
cd rssn-advanced

# 2. Build (JIT enabled by default)
cargo build

# 3. Run the test suite
cargo test --all

# 4. Build without JIT (for no-std / size-constrained targets)
cargo build --no-default-features
```

The build script (`build.rs`) runs `cbindgen` to regenerate the C header.
Set `DEV=1` to enable header generation during development:

```bash
DEV=1 cargo build
```

---

## 3. Development workflow

1. **Create a branch** from `main`:

   ```bash
   git checkout -b fix/my-bug
   # or
   git checkout -b feat/my-feature
   ```

2. **Make your changes.**  Add or update tests for any new behaviour.

3. **Format and lint** (mandatory — CI will reject failures):

   ```bash
   cargo +nightly fmt --all
   cargo clippy --all-targets --all-features -- -D warnings
   ```

4. **Run the full test suite:**

   ```bash
   cargo test --all
   cargo test --all --no-default-features   # also test without JIT
   ```

5. **Commit** with a conventional commit message:

   ```
   feat(jit): add AVX-512 batch path for f64×8
   fix(egraph): correct cost accounting for Neg nodes
   docs(custom): add C FFI usage example to module doc
   perf(asm_presets): use vfmadd for fused mul-add on AArch64
   ```

6. **Open a PR** against `main`.  Include a brief description of what
   changed and why.

---

## 4. Code style requirements

The crate enforces an **intentionally strict** lint configuration in
`src/lib.rs`.  Key rules:

| Rule | Level | Notes |
|------|-------|-------|
| `dead_code` | **deny** | Every public item must be reachable |
| `missing_docs` | **deny** | Every `pub` item needs a doc comment |
| `clippy::unwrap_used` / `expect_used` | **deny** | Use `?` or explicit error handling |
| `clippy::indexing_slicing` | **deny** | Use `get` / iterator methods |
| `clippy::arithmetic_side_effects` | **deny** | Use checked / saturating / wrapping arithmetic |
| `clippy::single_call_fn` | **deny** | Inline one-use helpers unless they improve readability |
| `unsafe_code` | **allow** | Required for asm presets and FFI; every `unsafe` block needs a `# Safety` comment |

**`asm_presets` additions** must include all three architecture paths
(x86_64, AArch64, riscv64) **and** a scalar fallback.  See
`src/asm_presets/add_f64x2_neon.rs` for the canonical pattern.

**FFI functions** must:
- return `RssnStatus` (not panic or abort on bad input);
- wrap the body in `catch_unwind(std::panic::AssertUnwindSafe(|| { … }))`;
- check all pointer arguments for null before dereferencing.

---

## 5. Testing

Tests live in three places:

| Location | What |
|----------|------|
| `src/**/*.rs` — `#[cfg(test)] mod tests` | Unit tests co-located with the code |
| `tests/` | Integration tests (Rust) |
| `tests/compare_sympy.py` | Python / NumPy accuracy and timing comparison |

**Run everything:**

```bash
cargo test --all                     # Rust tests, JIT enabled
cargo test --all --no-default-features  # Rust tests, no JIT

# Python comparison (requires numpy and sympy):
python tests/compare_sympy.py
```

**Property-based tests** use `proptest`.  If you add a new mathematical
transformation, consider adding a proptest that verifies it against a
reference implementation.

---

## 6. Benchmarking

Benchmarks use [Criterion](https://github.com/bheisler/criterion.rs):

```bash
cargo bench
```

If your change affects bulk evaluation, JIT compilation time, or
simplification throughput, please include before/after Criterion output
in your PR description.

**Benchmark environment notes:**

- Pin to a single core and disable Turbo Boost for reproducible results.
- Report the hardware spec (CPU model, RAM, OS) alongside numbers.
- The canonical comparison is against hand-optimised NumPy; see
  `tests/compare_sympy.py` and [`bench_reports.md`](bench_reports.md).

We do not accept performance claims without benchmark evidence.  We also do
not expect benchmark results to be perfect — honest "X% improvement on my
machine" is fine.

---

## 7. Pull request checklist

Before marking a PR ready for review, confirm:

- [ ] `cargo +nightly fmt --all` — no diff
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` — zero warnings
- [ ] `cargo test --all` — all tests pass
- [ ] `cargo test --all --no-default-features` — all tests pass without JIT
- [ ] New public items have doc comments (including `# Safety` for `unsafe fn`)
- [ ] New `asm_presets` entries cover all three architectures + scalar fallback
- [ ] New FFI functions use `catch_unwind(AssertUnwindSafe(...))` and null-check pointers
- [ ] `CHANGELOG` or PR description notes the change type (feat / fix / perf / docs)

---

## 8. Good first issues

Look for issues tagged
[`good first issue`](https://github.com/Apich-Organization/rssn-advanced/issues?q=is%3Aissue+label%3A%22good+first+issue%22)
on GitHub.  Typical entry points:

- Adding a missing doc comment or example to an existing public item.
- Extending an `asm_presets` file that only has one or two architectures.
- Adding a `proptest` for an existing transformation.
- Improving an error message in the parser.
- Fixing a Clippy lint that was temporarily `#[allow]`-ed with a `TODO`.

---

## 9. Communication

- **GitHub Issues** — bug reports, feature requests, design discussions.
- **GitHub Discussions** — open-ended questions and ideas.
- **Discord** — real-time chat: [discord.gg/D5e2czMTT9](https://discord.gg/D5e2czMTT9)
- **E-mail** — for security issues or private matters, contact the author
  directly: [Xinyu.Yang@apich.org](mailto:Xinyu.Yang@apich.org)
  (see also [SECURITY.md](SECURITY.md)).

Contributors are credited in release notes.  Thank you for helping make
rssn-advanced better!
