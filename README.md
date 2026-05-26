# rssn-advanced

[![Crates.io](https://img.shields.io/crates/v/rssn-advanced.svg)](https://crates.io/crates/rssn-advanced)
[![Docs.rs](https://docs.rs/rssn-advanced/badge.svg)](https://docs.rs/rssn-advanced)
[![License](https://img.shields.io/crates/l/rssn-advanced)](LICENSE)
[![Scc Count Badge Code](https://sloc.xyz/github/Apich-Organization/rssn-advanced/?category=code)](https://github.com/Apich-Organization/rssn-advanced/)
[![Scc Count Badge Lines](https://sloc.xyz/github/Apich-Organization/rssn-advanced/?category=lines)](https://github.com/Apich-Organization/rssn-advanced/)
[![Scc Count Badge Comments](https://sloc.xyz/github/Apich-Organization/rssn-advanced/?category=comments)](https://github.com/Apich-Organization/rssn-advanced/)
[![Discord Server](https://img.shields.io/discord/1459399539403522074.svg?label=Discord&logo=discord&color=blue)](https://discord.gg/D5e2czMTT9)
[![DOI](https://zenodo.org/badge/DOI/10.6084/m9.figshare.31044715.svg)](https://doi.org/10.6084/m9.figshare.31044715)

**rssn-advanced** is the symbolic computation engine of the [rssn](https://github.com/Apich-Organization/rssn) project.
It provides:

- a **hash-consed expression DAG** for structurally-shared, deduplicated storage of symbolic expressions;
- a **[Cranelift](https://cranelift.dev/)-backed JIT compiler** that turns a DAG subgraph into a native `f64` function at runtime, with a 2-row ILP batch path for higher throughput;
- **heuristic and e-graph simplification** — a rule-registry-driven greedy simplifier and a lightweight equality-saturation engine (no `egg` dependency);
- a **unified custom-operator API** that wires a user-defined function into the JIT, the simplifier, and the e-graph with a single descriptor;
- **hand-written inline-asm presets** for `f64×2` / `f64×4` arithmetic on x86_64 (SSE2 / AVX2 / AES-NI), AArch64 (NEON / crypto), and riscv64 (RVV 1.0 / Zkn);
- a **flat `extern "C"` API** compatible with `cbindgen` for embedding in C, C++, Python (ctypes/cffi), or any other language.

---

## When to use rssn-advanced

| Use case | Good fit? |
|----------|:---------:|
| Evaluate the same symbolic expression over millions of rows | ✅ |
| Symbolically simplify / rewrite algebraic expressions | ✅ |
| Embed a fast expression evaluator in a C/C++ application | ✅ |
| Runtime-define custom mathematical operators | ✅ |
| Fixed-width SIMD kernels (f64×2, f64×4) without JIT overhead | ✅ |
| GPU-accelerated batch evaluation | ❌ (not yet) |
| BLAS / LAPACK matrix operations | ❌ (out of scope) |

---

## Performance

Bulk evaluation of **N = 1,000,000 rows**, best of 5 runs.

**Test hardware:** Dell Latitude 5400 · Intel i7-8665U @ 1.90 GHz · 32 GiB RAM · Fedora Linux 44, kernel 6.19
(laptop-class CPU; server CPUs with larger L3 caches will show a smaller gap for simple expressions).

**Baseline:** hand-optimised NumPy (BLAS-linked, SIMD-enabled C backend).

```text
Expression                                      JIT bulk    JIT batch    NumPy     bulk/batch speedup
──────────────────────────────────────────────  ────────    ─────────    ─────     ────────────────
x + y + 10.0          (trivial baseline)         1.87 ns     1.13 ns    2.82 ns    1.5× /  2.5×
(x-y)^4               (degree-4 polynomial)      2.73 ns     1.28 ns   19.21 ns    7×  / 15×
cubic surface          (10 terms, 3 vars)         3.73 ns     1.77 ns   75.75 ns   20×  / 43×
rational w/ CSE        (repeated subexpression)  2.53 ns     1.27 ns   15.91 ns    6×  / 13×
```

**Why the gap grows with expression complexity:**
NumPy allocates one `float64[N]` scratch array per arithmetic operation.
A 10-term expression at N = 10⁶ creates ~200 MB of temporaries that
overflow L3 cache.  The JIT keeps every intermediate value in a CPU
register across the full expression, paying exactly one memory round-trip
per input column — 0 intermediate arrays regardless of expression depth.

**Honest caveats:**
- Numbers from a single laptop; your mileage will vary.
- The JIT batch path uses **2-row ILP unrolling**, not AVX-width vector instructions.
  For peak throughput on AVX-capable CPUs, use the [`asm_presets`] / [`simd`] paths for fixed kernels.
- Cranelift's register allocator produces good but not hand-tuned code.
  `gcc -O3` or LLVM can generate tighter loops for very simple expressions.

Full benchmark report: [`bench_reports.md`](bench_reports.md).

---

## Quick start

### Rust

Add to `Cargo.toml`:

```toml
[dependencies]
rssn-advanced = "0.1"
```

Parse an expression and JIT-compile it:

```rust
# // cfg guard: skip compilation when cranelift-jit feature is absent
# #[cfg(not(feature = "cranelift-jit"))] fn main() {}
# #[cfg(feature = "cranelift-jit")] fn main() {
use rssn_advanced::dag::builder::DagBuilder;
use rssn_advanced::parser::expr::parse_expression;
use rssn_advanced::ast::convert::dag_to_ast;
use rssn_advanced::jit::compiler::JitCompiler;

let mut builder = DagBuilder::new();
let root = parse_expression("x^2 + 2*x + 1", &mut builder).unwrap();

let mut compiler = JitCompiler::try_new().unwrap();
let ast = dag_to_ast(builder.arena(), root);
let f   = compiler.compile(&ast).unwrap();

// CompiledExprFn = extern "C" fn(*const f64) -> f64
let args = [3.0_f64];
assert_eq!(f(args.as_ptr()), 16.0);  // (3+1)^2

// 2-row ILP batch path (returns None if expression is not vectorizable)
let _batch = compiler.compile_batch_f64x2(&ast).unwrap();
# }
```

Register a custom operator that plugs into all pipeline stages:

```rust
# // cfg guard: skip compilation when cranelift-jit feature is absent
# #[cfg(not(feature = "cranelift-jit"))] fn main() {}
# #[cfg(feature = "cranelift-jit")] fn main() {
use std::sync::Arc;
use rssn_advanced::dag::builder::DagBuilder;
use rssn_advanced::custom::descriptor::{CustomOpDescriptor, CustomOpRegistry, EvalFn};
use rssn_advanced::egraph::egraph::{EGraph, EGraphConfig};
use rssn_advanced::jit::compiler::JitCompiler;

extern "C" fn relu(x: f64) -> f64 { x.max(0.0) }

let mut builder = DagBuilder::new();
// intern_function returns the FnId for this operator (field is crate-private)
let fn_id = builder.intern_function("relu");

let desc = CustomOpDescriptor::builder(fn_id, "relu", EvalFn::Arity1(relu))
    .vectorizable()   // safe to duplicate in the ILP batch path
    .cost(1.0)
    .build();

let mut reg = CustomOpRegistry::new();
reg.register(desc).unwrap();
let reg = Arc::new(reg);

// Wire into the JIT (registers the eval_fn pointer)
let mut compiler = JitCompiler::try_new().unwrap();
reg.apply_to_jit(&mut compiler);

// Wire into the heuristic simplifier
let _rule_reg = reg.build_rule_registry();

// Wire into the e-graph saturation engine
{
    let mut egraph = EGraph::new(&mut builder, EGraphConfig::default());
    reg.apply_to_egraph(&mut egraph);
}
# }
```

### C / C++

```c
/*
 * basics.c — rssn-advanced C API walkthrough
 *
 * Demonstrates:
 *   1. Building a DAG expression (x^2 + 2*x + 1)
 *   2. Simplifying it with the heuristic engine
 *   3. JIT-compiling and evaluating it
 *   4. Registering a custom operator (relu) and evaluating relu(x+3)
 *
 * Build:  make -C examples all
 * Run:    make -C examples run
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "../rssn-advanced.h"

/* ── helpers ──────────────────────────────────────────────────────────────── */

#define CHECK(call, msg)                                        \
    do {                                                        \
        enum RssnStatus _s = (call);                           \
        if (_s != RssnStatusSuccess) {                          \
            fprintf(stderr, "FAIL [status=%d]: %s\n", _s, msg); \
            exit(1);                                            \
        }                                                       \
    } while (0)

/* Simple relu — must have C linkage so the JIT can emit a direct call. */
static double relu_impl(double x) { return x > 0.0 ? x : 0.0; }

/* ── main ─────────────────────────────────────────────────────────────────── */

int main(void)
{
    /* ── 1. Build the expression: x^2 + 2*x + 1 ─────────────────────────── */
    struct DagBuilder *dag = rssn_dag_new();
    if (!dag) { fputs("rssn_dag_new failed\n", stderr); return 1; }

    uint32_t x_id, two_id, one_id;
    uint32_t x2_id, two_x_id, sum1_id, root_id;

    CHECK(rssn_dag_variable_v2(dag, "x", &x_id),       "variable x");
    CHECK(rssn_dag_constant_v2(dag, 2.0, &two_id),     "constant 2");
    CHECK(rssn_dag_constant_v2(dag, 1.0, &one_id),     "constant 1");
    CHECK(rssn_dag_pow_v2    (dag, x_id, two_id, &x2_id),    "x^2");
    CHECK(rssn_dag_mul_v2    (dag, two_id, x_id, &two_x_id), "2*x");
    CHECK(rssn_dag_add_v2    (dag, x2_id, two_x_id, &sum1_id), "x^2+2x");
    CHECK(rssn_dag_add_v2    (dag, sum1_id, one_id, &root_id), "x^2+2x+1");

    printf("Expression built.  Root node id = %u\n", root_id);

    /* ── 2. JIT compile ──────────────────────────────────────────────────── */
    void *fn_ptr = NULL;
    CHECK(rssn_dag_compile_v2(dag, root_id, &fn_ptr), "compile x^2+2x+1");

    /* ── 3. Evaluate at several x values ────────────────────────────────── */
    printf("\nx^2 + 2x + 1  (should equal (x+1)^2):\n");
    double xs[] = { -2.0, -1.0, 0.0, 1.0, 2.0, 3.0 };
    for (int i = 0; i < 6; i++) {
        double result = 0.0;
        double vars[] = { xs[i] };
        CHECK(rssn_dag_execute_v2(fn_ptr, vars, &result), "execute");
        printf("  f(%.1f) = %.1f   expected %.1f%s\n",
               xs[i], result, (xs[i] + 1.0) * (xs[i] + 1.0),
               result == (xs[i]+1.0)*(xs[i]+1.0) ? " ✓" : " ✗");
    }

    /* ── 4. Custom operator: relu ────────────────────────────────────────── */
    printf("\nCustom operator  relu(x + 3):\n");

    struct RssnCustomOpRegistry *reg = rssn_custom_op_registry_new();
    if (!reg) { fputs("registry_new failed\n", stderr); return 1; }

    /* Build relu(x + 3) first so we can intern the function name and get
     * the stable FnId assigned by the DagBuilder.  That same FnId must be
     * used when registering the eval_fn pointer in the custom-op registry. */
    struct DagBuilder *dag2 = rssn_dag_new();
    uint32_t x2, three_id, xp3_id, relu_fn_id, relu_node;
    CHECK(rssn_dag_variable_v2(dag2, "x",    &x2),       "variable x");
    CHECK(rssn_dag_constant_v2(dag2, 3.0,    &three_id), "constant 3");
    CHECK(rssn_dag_add_v2     (dag2, x2, three_id, &xp3_id), "x+3");

    /* Intern "relu" in dag2 — this allocates the FnId we must use everywhere. */
    relu_fn_id = rssn_dag_intern_function(dag2, "relu");
    if (relu_fn_id == (uint32_t)-1) {
        fputs("rssn_dag_intern_function failed\n", stderr); return 1;
    }

    /* Register the eval pointer under the SAME FnId that the builder assigned. */
    CHECK(rssn_custom_op_register_fn1(reg, relu_fn_id, "relu",
                                      (double(*)(double))relu_impl,
                                      /*vectorizable=*/1),
          "register relu");

    CHECK(rssn_dag_call_fn_v2 (dag2, relu_fn_id, &xp3_id, 1, &relu_node),
          "relu(x+3)");

    void *relu_fn = NULL;
    CHECK(rssn_dag_compile_with_custom_ops(dag2, relu_node, reg, &relu_fn),
          "compile relu(x+3)");

    double test_xs[] = { -5.0, -3.0, -1.0, 0.0, 2.0 };
    for (int i = 0; i < 5; i++) {
        double result = 0.0, expected;
        double v[] = { test_xs[i] };
        CHECK(rssn_dag_execute_v2(relu_fn, v, &result), "execute relu");
        expected = (test_xs[i] + 3.0) > 0.0 ? (test_xs[i] + 3.0) : 0.0;
        printf("  relu(%.1f + 3) = %.1f   expected %.1f%s\n",
               test_xs[i], result, expected,
               result == expected ? " ✓" : " ✗");
    }

    /* ── cleanup ─────────────────────────────────────────────────────────── */
    rssn_custom_op_registry_free(reg);
    rssn_dag_free(dag2);
    rssn_dag_free(dag);

    puts("\nDone.");
    return 0;
}
```

---

## Architecture overview

```text
┌─────────────────────────────────────────────────────────┐
│                     rssn-advanced                       │
│                                                         │
│  parser ──→ dag ──→ ast ──→ jit ──→ native fn ptr      │
│               │       │       │                         │
│               │       └──→ heuristic simplifier         │
│               │       └──→ e-graph saturation           │
│               │                                         │
│               └──→ custom (one descriptor → all stages) │
│                                                         │
│  asm_presets / simd  (fixed-width f64×2 / f64×4)        │
│  ffi  (extern "C", cbindgen-compatible)                  │
│  parallel / storage / runtime  (infrastructure)          │
└─────────────────────────────────────────────────────────┘
```

| Module | Role |
|--------|------|
| `dag` | Hash-consed node store — structural sharing, deduplication |
| `ast` | Local tree projection (relative `i32` pointers) for algorithm traversal |
| `parser` | `nom`-based infix parser: variables, constants, `+−×÷^%`, named functions |
| `jit` | Cranelift JIT: scalar + 2-row ILP batch compilation |
| `heuristic` | Greedy/beam simplifier with pluggable rule registries |
| `egraph` | Union-find equality saturation with cost-based extraction |
| `custom` | Unified custom-operator descriptor + registry |
| `simd` | Slice-level wrappers over `asm_presets` |
| `asm_presets` | `f64×2` / `f64×4` inline-asm for x86_64 / AArch64 / riscv64 |
| `ffi` | Flat `extern "C"` API + async bridge (fiber-backed via `dtact`) |
| `parallel` | Fiber-based parallel simplification |
| `storage` | Disk-backed spill + hot-node frequency cache |

---

## Feature flags

| Flag | Default | Effect |
|------|:-------:|--------|
| `cranelift-jit` | **on** | Enables the `jit` module and all JIT / batch-compile paths |

Disable with `--no-default-features` for embedded or WASM targets.
The parser, DAG, simplifier, e-graph, and SIMD presets remain available without JIT.

---

## Known limitations

- **Parser coverage:** arithmetic operators (`+`, `-`, `*`, `/`, `^`, `%`), unary negation, and named functions.  Transcendentals (`sin`, `exp`, …) must be registered as custom operators.
- **JIT batch = 2-row ILP only:** the batch path does not emit wide SIMD (AVX2 / AVX-512) instructions.  Use `asm_presets` / `simd` for fixed-width high-throughput kernels.
- **Single-threaded JIT context:** concurrent compilation requests serialise on a global `Mutex`.
- **No GPU support.**
- **e-graph extractor is greedy:** the bottom-up cost minimiser is fast but not globally optimal (optimal extraction is NP-hard).
- **Windows support is experimental:** some inline-asm presets fall through to the scalar path on Windows; correctness is maintained.

---

## Contributing

We welcome bug reports, performance improvements, new features, and documentation fixes.
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code-style requirements, and the PR workflow.

---

## Maintainers & Contributors

- **Author:** [Pana Yang](https://github.com/panayang) (ORCID: 0009-0007-2600-0948 · [Xinyu.Yang@apich.org](mailto:Xinyu.Yang@apich.org))
- **Consultants:**
  - X. Zhang — Algorithm & Informatics ([@RheaCherry](https://github.com/RheaCherry) · [3248998213@qq.com](mailto:3248998213@qq.com))
  - Z. Wang — Mathematics
  - Y. Li — Physics ([xian1360685019@qq.com](mailto:xian1360685019@qq.com))
- **Project Reviewer:** Z. Li

---

## License

Licensed under the [Apache 2.0 License](LICENSE).

---

## Related

- [SECURITY.md](SECURITY.md) — responsible disclosure policy
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) — community standards
- [bench_reports.md](bench_reports.md) — full benchmark output
- [GitHub Wiki](https://github.com/Apich-Organization/rssn-advanced/wiki) — extended documentation
