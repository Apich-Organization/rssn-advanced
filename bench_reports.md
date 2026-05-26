# Bench Reports for rssn-advanced (2026-05-26)

Runs on:
```text
Operating System: Fedora Linux 44
KDE Plasma Version: 6.6.4
KDE Frameworks Version: 6.25.0
Qt Version: 6.10.3
Kernel Version: 6.19.14-300.fc44.x86_64 (64-bit)
Graphics Platform: Wayland
Processors: 8 × Intel® Core™ i7-8665U CPU @ 1.90GHz
Memory: 32 GiB of RAM (31.0 GiB usable)
Graphics Processor: Intel® UHD Graphics 620
Manufacturer: Dell Inc.
Product Name: Latitude 5400
```

Command:
```bash
python ./tests/compare_sympy.py 
```

Results:
```text
==============================================================================
   RSSN-Advanced JIT vs NumPy — Bulk Evaluation Benchmark
   N = 1,000,000 rows per expression  |  5 repeats, best time reported
==============================================================================

──────────────────────────────────────────────────────────────────────────────
  1. Trivial (baseline)
  x + y + 10.0
──────────────────────────────────────────────────────────────────────────────
  Rust JIT bulk  (scalar, Rust loop)              1.868 ms     1.87 ns/eval
  Rust JIT batch (2-row ILP vectorised)           1.133 ms     1.13 ns/eval
  NumPy (SIMD / C, hand-optimised)                2.824 ms     2.82 ns/eval
  SymPy lambdify → numpy backend                  2.763 ms     2.76 ns/eval

  JIT bulk  vs NumPy:  1.51x faster
  JIT batch vs NumPy:  2.49x faster

  Accuracy  bulk  max|Δ|=0.00e+00  ✔
            batch max|Δ|=0.00e+00  ✔

  NumPy intermediate arrays: ~2 ops → ~15 MB peak temp memory
  JIT: 0 intermediate arrays — all values kept in CPU registers

──────────────────────────────────────────────────────────────────────────────
  2. Degree-4 polynomial  (x-y)^4  [2 vars]
  x^4 - 4*x^3*y + 6*x^2*y^2 - 4*x*y^3 + y^4
──────────────────────────────────────────────────────────────────────────────
  Rust JIT bulk  (scalar, Rust loop)              2.729 ms     2.73 ns/eval
  Rust JIT batch (2-row ILP vectorised)           1.276 ms     1.28 ns/eval
  NumPy (SIMD / C, hand-optimised)               19.207 ms    19.21 ns/eval
  SymPy lambdify → numpy backend                 18.933 ms    18.93 ns/eval

  JIT bulk  vs NumPy:  7.04x faster
  JIT batch vs NumPy: 15.06x faster

  Accuracy  bulk  max|Δ|=5.46e-12  ✔
            batch max|Δ|=5.46e-12  ✔

  NumPy intermediate arrays: ~16 ops → ~122 MB peak temp memory
  JIT: 0 intermediate arrays — all values kept in CPU registers

──────────────────────────────────────────────────────────────────────────────
  3. Cubic surface  [3 vars, 10 terms]
  x^3 + y^3 + z^3 - 3*x*y*z + x^2*y - x*y^2 + y^2*z - y*z^2 + z^2*x - z*x^2
──────────────────────────────────────────────────────────────────────────────
  Rust JIT bulk  (scalar, Rust loop)              3.733 ms     3.73 ns/eval
  Rust JIT batch (2-row ILP vectorised)           1.772 ms     1.77 ns/eval
  NumPy (SIMD / C, hand-optimised)               75.751 ms    75.75 ns/eval
  SymPy lambdify → numpy backend                 78.214 ms    78.21 ns/eval

  JIT bulk  vs NumPy: 20.29x faster
  JIT batch vs NumPy: 42.74x faster

  Accuracy  bulk  max|Δ|=2.84e-13  ✔
            batch max|Δ|=2.84e-13  ✔

  NumPy intermediate arrays: ~27 ops → ~206 MB peak temp memory
  JIT: 0 intermediate arrays — all values kept in CPU registers

──────────────────────────────────────────────────────────────────────────────
  4. Rational w/ CSE  [2 vars, repeated subexpr]
  (x^2 + y^2) / (x^2 + y^2 + 1.0) + x*y*(x^2 - y^2) / (x^2 + y^2 + 1.0)^2
──────────────────────────────────────────────────────────────────────────────
  Rust JIT bulk  (scalar, Rust loop)              2.534 ms     2.53 ns/eval
  Rust JIT batch (2-row ILP vectorised)           1.266 ms     1.27 ns/eval
  NumPy (SIMD / C, hand-optimised)               15.913 ms    15.91 ns/eval
  SymPy lambdify → numpy backend                 22.961 ms    22.96 ns/eval

  JIT bulk  vs NumPy:  6.28x faster
  JIT batch vs NumPy: 12.57x faster

  Accuracy  bulk  max|Δ|=0.00e+00  ✔
            batch max|Δ|=0.00e+00  ✔

  NumPy intermediate arrays: ~20 ops → ~153 MB peak temp memory
  JIT: 0 intermediate arrays — all values kept in CPU registers

==============================================================================
  SUMMARY: JIT speedup vs hand-optimised NumPy
  Expression                                          bulk     batch
  ──────────────────────────────────────────────  ────────  ────────
  1. Trivial (baseline)                             1.51x    2.49x
  2. Degree-4 polynomial                            7.04x   15.06x
  3. Cubic surface                                 20.29x   42.74x
  4. Rational w/ CSE                                6.28x   12.57x

  Observation: speedup grows with expression complexity because
  NumPy's intermediate arrays overflow L2/L3 cache at N=1,000,000.
  JIT maintains register-resident computation across the entire
  expression, paying one memory read/write per input element.
==============================================================================
```
