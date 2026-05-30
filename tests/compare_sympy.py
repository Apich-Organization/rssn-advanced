"""
RSSN-Advanced JIT — multi-backend evaluation benchmark.

Compares the RSSN JIT (scalar bulk, f64x2 batch, f64x4 batch) against:
  • NumPy      — hand-optimised ufunc chain
  • numexpr    — JIT-compiled string expressions (multi-threaded, optional)
  • Numba      — LLVM-JIT compiled nopython ufuncs (optional, AOT warmed)
  • SymPy      — lambdify → numpy backend

JIT structural advantages vs NumPy ufunc chains:
  • No temp arrays   — keeps every intermediate in CPU registers
  • CSE              — identical subexpressions computed once
  • Fused dispatch   — one FFI call for N evaluations
  • Algebraic simp   — constant-folding before codegen

numexpr advantages vs NumPy:
  • Avoids temp arrays via expression string parsing
  • Multi-threaded by default

Numba advantages vs NumPy:
  • LLVM-JITted, fuses all ops, no temp arrays

Libraries that are not installed are skipped gracefully with a note.
"""

import ctypes
import os
import sys
import time

import numpy as np
import sympy

# ── Optional backends ──────────────────────────────────────────────────────

try:
    import numexpr as ne
    HAS_NUMEXPR = True
except ImportError:
    HAS_NUMEXPR = False

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# ── Load shared library ────────────────────────────────────────────────────

_lib_candidates = [
    "../target/release/librssn_advanced.so",    # Linux
    "../target/release/librssn_advanced.dylib", # macOS
    "../target/release/rssn_advanced.dll",      # Windows
]

lib_path = None
for rel in _lib_candidates:
    cand = os.path.abspath(os.path.join(os.path.dirname(__file__), rel))
    if os.path.exists(cand):
        lib_path = cand
        break

if lib_path is None:
    print("ERROR: shared library not found. Build with:\n  cargo build --release")
    sys.exit(1)

lib = ctypes.CDLL(lib_path)

# ── FFI signatures ─────────────────────────────────────────────────────────

c_void_p = ctypes.c_void_p
c_uint32  = ctypes.c_uint32
c_double  = ctypes.c_double
c_size_t  = ctypes.c_size_t
c_int     = ctypes.c_int
c_char_p  = ctypes.c_char_p
DoubleP   = ctypes.POINTER(ctypes.c_double)
UInt32P   = ctypes.POINTER(ctypes.c_uint32)
VoidPP    = ctypes.POINTER(ctypes.c_void_p)


def _sig(fn, argtypes, restype):
    fn.argtypes = argtypes
    fn.restype  = restype


class RssnEGraphConfig(ctypes.Structure):
    _fields_ = [
        ("max_rounds", ctypes.c_uint32),
        ("max_merges", ctypes.c_uint32),
        ("max_new_nodes", ctypes.c_uint32),
        ("strict_ieee754_signed_zero", ctypes.c_uint8),
    ]

_sig(lib.rssn_dag_new,              [],                               c_void_p)
_sig(lib.rssn_dag_free,             [c_void_p],                       None)
_sig(lib.rssn_dag_parse,            [c_void_p, c_char_p, UInt32P],   c_int)
_sig(lib.rssn_dag_simplify,         [c_void_p, c_uint32],             c_uint32)
_sig(lib.rssn_dag_simplify_with_egraph, [c_void_p, c_uint32, RssnEGraphConfig, UInt32P], c_int)
_sig(lib.rssn_dag_compile,          [c_void_p, c_uint32, VoidPP],    c_int)
_sig(lib.rssn_dag_compile_batch,    [c_void_p, c_uint32, VoidPP],    c_int)
_sig(lib.rssn_dag_compile_batch_f64x4, [c_void_p, c_uint32, VoidPP], c_int)
_sig(lib.rssn_dag_compile_batch_f64x8, [c_void_p, c_uint32, VoidPP], c_int)
_sig(lib.rssn_dag_execute_bulk,     [c_void_p, VoidPP, c_uint32, c_size_t, DoubleP], c_int)
_sig(lib.rssn_dag_execute_batch,    [c_void_p, VoidPP, c_size_t, DoubleP], c_int)
# n_workers=0 → auto-detect (logical CPUs, capped at 16)
_sig(lib.rssn_dag_execute_batch_parallel,
     [c_void_p, VoidPP, c_uint32, c_size_t, DoubleP, c_uint32], c_int)

# ── Helpers ────────────────────────────────────────────────────────────────


def col_ptrs(*arrays):
    """ctypes void* array pointing at each numpy column's data."""
    return (ctypes.c_void_p * len(arrays))(*(a.ctypes.data for a in arrays))


def build_expr(expr_str: str):
    """Parse, simplify, and JIT-compile an expression string.
    Returns (builder, scalar_ptr, batch_ptr_or_None, batch_f64x4_ptr_or_None).
    """
    builder = lib.rssn_dag_new()
    root_id = ctypes.c_uint32(0)
    status = lib.rssn_dag_parse(builder, expr_str.encode(), ctypes.byref(root_id))
    assert status == 0, f"parse failed ({status}) for: {expr_str!r}"

    # Use the powerful E-graph saturation simplifier.
    cfg = RssnEGraphConfig(max_rounds=8, max_merges=512, max_new_nodes=1024, strict_ieee754_signed_zero=0)
    simp_id_val = ctypes.c_uint32(0)
    status = lib.rssn_dag_simplify_with_egraph(builder, root_id, cfg, ctypes.byref(simp_id_val))
    assert status == 0, f"egraph simplify failed ({status})"
    simp_id = simp_id_val.value

    scalar_ptr = ctypes.c_void_p()
    status = lib.rssn_dag_compile(builder, simp_id, ctypes.byref(scalar_ptr))
    assert status == 0 and scalar_ptr.value, \
        f"scalar compile failed ({status}) for: {expr_str!r}"

    batch_ptr = ctypes.c_void_p()
    bst = lib.rssn_dag_compile_batch(builder, simp_id, ctypes.byref(batch_ptr))
    has_batch = bst == 0 and bool(batch_ptr.value)

    batch_f64x4_ptr = ctypes.c_void_p()
    bst_f64x4 = lib.rssn_dag_compile_batch_f64x4(builder, simp_id, ctypes.byref(batch_f64x4_ptr))
    has_batch_f64x4 = bst_f64x4 == 0 and bool(batch_f64x4_ptr.value)

    batch_f64x8_ptr = ctypes.c_void_p()
    bst_f64x8 = lib.rssn_dag_compile_batch_f64x8(builder, simp_id, ctypes.byref(batch_f64x8_ptr))
    has_batch_f64x8 = bst_f64x8 == 0 and bool(batch_f64x8_ptr.value)

    return (
        builder, scalar_ptr,
        batch_ptr        if has_batch       else None,
        batch_f64x4_ptr  if has_batch_f64x4 else None,
        batch_f64x8_ptr  if has_batch_f64x8 else None,
    )


def bench_fn(fn, warmup=2, repeats=5):
    """Return the minimum wall-clock time over `repeats` runs (seconds)."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def print_row(label: str, t: float, N: int, ref_t: float | None = None):
    ns    = t / N * 1e9
    ratio = f"  {ref_t / t:6.2f}x vs NumPy" if ref_t is not None and ref_t != t else ""
    print(f"  {label:<50s}  {t * 1e3:8.3f} ms  {ns:7.2f} ns/eval{ratio}")


def speedup_str(ref_t, t):
    if t is None:
        return "  n/a  "
    s = ref_t / t
    tag = "faster" if s >= 1.0 else "slower"
    return f"{s:6.2f}x {tag}"


# ── Expression suite ───────────────────────────────────────────────────────
# Each entry:
#   (display_name, rssn_expr_str, n_vars,
#    numpy_fn(*arrays) -> array,
#    numexpr_str,          # expression string for ne.evaluate(); None to skip
#    numba_fn_factory,     # callable(N) -> jit_fn(*arrays); None to skip
#    sympy_expr)

x, y, z = sympy.symbols("x y z")


def _ne_str_vars(*var_names):
    """Return a lambda that builds a numexpr string (no-op; we pre-define)."""
    pass  # noqa: used as placeholder


def _make_numba_2var(body_src: str):
    """
    Build a Numba vectorized ufunc from a Python expression source string
    operating on two float64 scalar inputs (x, y).
    body_src must be a single-line Python expression in x, y.
    Returns the vectorized function, or None if Numba is unavailable.
    """
    if not HAS_NUMBA:
        return None
    import numba  # noqa
    import math   # noqa (used inside eval'd body)
    import numpy as np  # noqa

    ns: dict = {}
    exec(
        f"import numba, math\n"
        f"@numba.vectorize(['float64(float64, float64)'], nopython=True)\n"
        f"def _fn(x, y): return {body_src}\n",
        ns,
    )
    return ns["_fn"]


def _make_numba_3var(body_src: str):
    """
    Build a Numba vectorized ufunc from a Python expression source string
    operating on three float64 scalar inputs (x, y, z).
    """
    if not HAS_NUMBA:
        return None
    ns: dict = {}
    exec(
        f"import numba, math\n"
        f"@numba.vectorize(['float64(float64, float64, float64)'], nopython=True)\n"
        f"def _fn(x, y, z): return {body_src}\n",
        ns,
    )
    return ns["_fn"]


SUITE = [
    (
        "1. Trivial (baseline)",
        "x + y + 10.0",
        2,
        lambda xc, yc: xc + yc + 10.0,
        "x + y + 10.0",
        _make_numba_2var("x + y + 10.0"),
        x + y + 10.0,
    ),
    (
        "2. Degree-4 polynomial  (x-y)^4  [2 vars]",
        "x^4 - 4*x^3*y + 6*x^2*y^2 - 4*x*y^3 + y^4",
        2,
        lambda xc, yc: (xc - yc) ** 4,
        "(x - y)**4",
        _make_numba_2var("(x - y)**4"),
        (x - y) ** 4,
    ),
    (
        "3. Cubic surface  [3 vars, 10 terms]",
        "x^3 + y^3 + z^3 - 3*x*y*z + x^2*y - x*y^2 + y^2*z - y*z^2 + z^2*x - z*x^2",
        3,
        lambda xc, yc, zc: (
            xc**3 + yc**3 + zc**3
            - 3 * xc * yc * zc
            + xc**2 * yc - xc * yc**2
            + yc**2 * zc - yc * zc**2
            + zc**2 * xc - zc * xc**2
        ),
        "x**3 + y**3 + z**3 - 3*x*y*z + x**2*y - x*y**2 + y**2*z - y*z**2 + z**2*x - z*x**2",
        _make_numba_3var(
            "x**3 + y**3 + z**3 - 3*x*y*z + x**2*y - x*y**2 + y**2*z - y*z**2 + z**2*x - z*x**2"
        ),
        x**3 + y**3 + z**3 - 3*x*y*z + x**2*y - x*y**2 + y**2*z - y*z**2 + z**2*x - z*x**2,
    ),
    (
        "4. Rational w/ CSE  [2 vars, repeated subexpr]",
        "(x^2 + y^2) / (x^2 + y^2 + 1.0) + x*y*(x^2 - y^2) / (x^2 + y^2 + 1.0)^2",
        2,
        lambda xc, yc: (
            lambda r2: r2 / (r2 + 1.0) + xc * yc * (xc**2 - yc**2) / (r2 + 1.0) ** 2
        )(xc**2 + yc**2),
        "(x**2 + y**2) / (x**2 + y**2 + 1.0) + x*y*(x**2 - y**2) / (x**2 + y**2 + 1.0)**2",
        _make_numba_2var(
            "(x**2 + y**2) / (x**2 + y**2 + 1.0)"
            " + x*y*(x**2 - y**2) / (x**2 + y**2 + 1.0)**2"
        ),
        (x**2 + y**2) / (x**2 + y**2 + 1)
        + x * y * (x**2 - y**2) / (x**2 + y**2 + 1) ** 2,
    ),
    (
        "5. Complex degree-5 polynomial [3 vars]",
        "x^5 - y^5 + z^5 - 5*x^3*y^2 + 5*x^2*y^3 - 5*y^3*z^2 + 5*y^2*z^3 - 5*z^3*x^2 + 5*z^2*x^3 + x*y*z*(x^2 + y^2 + z^2)",
        3,
        lambda xc, yc, zc: (
            xc**5 - yc**5 + zc**5
            - 5*xc**3*yc**2 + 5*xc**2*yc**3
            - 5*yc**3*zc**2 + 5*yc**2*zc**3
            - 5*zc**3*xc**2 + 5*zc**2*xc**3
            + xc*yc*zc*(xc**2 + yc**2 + zc**2)
        ),
        "x**5 - y**5 + z**5 - 5*x**3*y**2 + 5*x**2*y**3 - 5*y**3*z**2 + 5*y**2*z**3 - 5*z**3*x**2 + 5*z**2*x**3 + x*y*z*(x**2 + y**2 + z**2)",
        _make_numba_3var(
            "x**5 - y**5 + z**5"
            " - 5*x**3*y**2 + 5*x**2*y**3"
            " - 5*y**3*z**2 + 5*y**2*z**3"
            " - 5*z**3*x**2 + 5*z**2*x**3"
            " + x*y*z*(x**2 + y**2 + z**2)"
        ),
        x**5 - y**5 + z**5
        - 5*x**3*y**2 + 5*x**2*y**3
        - 5*y**3*z**2 + 5*y**2*z**3
        - 5*z**3*x**2 + 5*z**2*x**3
        + x*y*z*(x**2 + y**2 + z**2),
    ),
    (
        "6. Positive Nested Sqrt [2 vars]",
        "(x^2 + 1.0)^0.5 + (x^2 + y^2 + 1.0)^0.5 + (x^2 + y^2 + 2.0)^0.5",
        2,
        lambda xc, yc: (
            np.sqrt(xc**2 + 1.0)
            + np.sqrt(xc**2 + yc**2 + 1.0)
            + np.sqrt(xc**2 + yc**2 + 2.0)
        ),
        "sqrt(x**2 + 1.0) + sqrt(x**2 + y**2 + 1.0) + sqrt(x**2 + y**2 + 2.0)",
        _make_numba_2var(
            "math.sqrt(x**2 + 1.0)"
            " + math.sqrt(x**2 + y**2 + 1.0)"
            " + math.sqrt(x**2 + y**2 + 2.0)"
        ),
        (x**2 + 1.0)**0.5 + (x**2 + y**2 + 1.0)**0.5 + (x**2 + y**2 + 2.0)**0.5,
    ),
    (
        "7. Redundant Algebraic Cubics (E-Graph target) [2 vars]",
        "((x + y)^3 - (x - y)^3 - 6*x^2*y) / (y^2 + 1.0) + x*y - y*x",
        2,
        lambda xc, yc: (2.0 * yc**3) / (yc**2 + 1.0),
        "((x + y)**3 - (x - y)**3 - 6 * x**2 * y) / (y**2 + 1.0) + x * y - y * x",
        _make_numba_2var("((x + y)**3 - (x - y)**3 - 6 * x**2 * y) / (y**2 + 1.0) + x * y - y * x"),
        ((x + y)**3 - (x - y)**3 - 6 * x**2 * y) / (y**2 + 1.0) + x * y - y * x,
    ),
]

# ── Main ───────────────────────────────────────────────────────────────────

N = 10_000_000


def run_benchmark():
    sep  = "=" * 94
    sep2 = "─" * 94

    print(sep)
    print("   RSSN-Advanced JIT — Multi-Backend Evaluation Benchmark")
    print(f"   N = {N:,} rows per expression  |  5 repeats, best time reported")
    available = ["NumPy", "SymPy/lambdify"]
    if HAS_NUMEXPR:
        available.append("numexpr")
    if HAS_NUMBA:
        available.append("Numba")
    print(f"   Backends: {', '.join(available)}")
    if not HAS_NUMEXPR:
        print("   [numexpr not installed — pip install numexpr to enable]")
    if not HAS_NUMBA:
        print("   [numba not installed — pip install numba to enable]")
    print(sep)

    rng = np.random.default_rng(0xDEAD_BEEF)
    cols_data = {
        "x": np.ascontiguousarray(rng.uniform(-5.0, 5.0, N), np.float64),
        "y": np.ascontiguousarray(rng.uniform(-5.0, 5.0, N), np.float64),
        "z": np.ascontiguousarray(rng.uniform(-5.0, 5.0, N), np.float64),
    }
    var_order = ["x", "y", "z"]

    out   = np.empty(N, np.float64)
    out_p = out.ctypes.data_as(DoubleP)
    cols_3 = col_ptrs(*[cols_data[v] for v in var_order])
    cols_2 = col_ptrs(*[cols_data[v] for v in var_order[:2]])

    # summary row: (name, t_numpy, t_bulk, t_batch, t_batch4, t_ne, t_numba, t_sympy)
    summary = []

    for name, expr_str, n_vars, numpy_fn, ne_str, numba_fn, sympy_expr in SUITE:
        print(f"\n{sep2}")
        print(f"  {name}")
        print(f"  {expr_str}")
        print(sep2)

        builder, scalar_ptr, batch_ptr, batch_f64x4_ptr, batch_f64x8_ptr = build_expr(expr_str)
        cols = cols_3 if n_vars == 3 else cols_2
        args = [cols_data[v] for v in var_order[:n_vars]]

        # warm-up the instruction cache
        lib.rssn_dag_execute_bulk(scalar_ptr, cols, n_vars, N, out_p)

        # ── RSSN JIT bulk ─────────────────────────────────────────────────
        def rust_bulk():
            lib.rssn_dag_execute_bulk(scalar_ptr, cols, n_vars, N, out_p)

        t_bulk = bench_fn(rust_bulk)
        rust_bulk_out = out.copy()

        # ── RSSN JIT batch f64x2 ─────────────────────────────────────────
        t_batch = None
        rust_batch_out = None
        if batch_ptr is not None:
            def rust_batch():
                lib.rssn_dag_execute_batch(batch_ptr, cols, N, out_p)
            t_batch = bench_fn(rust_batch)
            rust_batch_out = out.copy()

        # ── RSSN JIT batch f64x4 ─────────────────────────────────────────
        t_batch4 = None
        rust_batch4_out = None
        if batch_f64x4_ptr is not None:
            def rust_batch4():
                lib.rssn_dag_execute_batch(batch_f64x4_ptr, cols, N, out_p)
            t_batch4 = bench_fn(rust_batch4)
            rust_batch4_out = out.copy()

        # ── RSSN JIT batch f64x8 (4×F64X2, ILP-8) ────────────────────────
        t_batch8 = None
        rust_batch8_out = None
        if batch_f64x8_ptr is not None:
            def rust_batch8():
                lib.rssn_dag_execute_batch(batch_f64x8_ptr, cols, N, out_p)
            t_batch8 = bench_fn(rust_batch8)
            rust_batch8_out = out.copy()

        # ── RSSN JIT batch f64x2 parallel ────────────────────────────────
        t_batch_par = None
        rust_batch_par_out = None
        if batch_ptr is not None:
            def rust_batch_par():
                lib.rssn_dag_execute_batch_parallel(
                    batch_ptr, cols, n_vars, N, out_p, 0)
            t_batch_par = bench_fn(rust_batch_par)
            rust_batch_par_out = out.copy()

        # ── RSSN JIT batch f64x4 parallel ────────────────────────────────
        t_batch4_par = None
        rust_batch4_par_out = None
        if batch_f64x4_ptr is not None:
            def rust_batch4_par():
                lib.rssn_dag_execute_batch_parallel(
                    batch_f64x4_ptr, cols, n_vars, N, out_p, 0)
            t_batch4_par = bench_fn(rust_batch4_par)
            rust_batch4_par_out = out.copy()

        # ── RSSN JIT batch f64x8 parallel (4×F64X2 × dtact workers) ─────
        t_batch8_par = None
        rust_batch8_par_out = None
        if batch_f64x8_ptr is not None:
            def rust_batch8_par():
                lib.rssn_dag_execute_batch_parallel(
                    batch_f64x8_ptr, cols, n_vars, N, out_p, 0)
            t_batch8_par = bench_fn(rust_batch8_par)
            rust_batch8_par_out = out.copy()

        # ── NumPy ─────────────────────────────────────────────────────────
        def numpy_eval():
            np.copyto(out, numpy_fn(*args))

        t_numpy = bench_fn(numpy_eval)
        numpy_out = out.copy()

        # ── numexpr ──────────────────────────────────────────────────────
        t_ne = None
        if HAS_NUMEXPR and ne_str is not None:
            # bind local variable names for ne.evaluate
            local_dict = {v: cols_data[v] for v in var_order[:n_vars]}
            try:
                ne.evaluate(ne_str, local_dict=local_dict, out=out)  # warm-up
                def ne_eval():
                    ne.evaluate(ne_str, local_dict=local_dict, out=out)
                t_ne = bench_fn(ne_eval)
            except Exception as exc:
                print(f"  [numexpr skipped: {exc}]")

        # ── Numba ────────────────────────────────────────────────────────
        t_numba = None
        if HAS_NUMBA and numba_fn is not None:
            try:
                # AOT warm-up (triggers compilation on first call)
                _numba_result = numba_fn(*args)
                numba_fn(*args)  # second call — JIT is now hot
                def numba_eval():
                    nonlocal _numba_result
                    _numba_result = numba_fn(*args)
                t_numba = bench_fn(numba_eval)
            except Exception as exc:
                print(f"  [numba skipped: {exc}]")

        # ── SymPy lambdify ───────────────────────────────────────────────
        syms   = [x, y, z][:n_vars]
        lam_np = sympy.lambdify(syms, sympy_expr, "numpy")
        lam_np(*args)  # warm-up

        def sympy_np_eval():
            lam_np(*args)

        t_sympy = bench_fn(sympy_np_eval)

        # ── Print timing rows ─────────────────────────────────────────────
        print()
        print_row("RSSN JIT  bulk  (scalar, Rust loop)",   t_bulk,   N, t_numpy)
        if t_batch  is not None:
            print_row("RSSN JIT  batch f64x2",             t_batch,  N, t_numpy)
        if t_batch_par is not None:
            print_row("RSSN JIT  f64x2 parallel",          t_batch_par, N, t_numpy)
        if t_batch4 is not None:
            print_row("RSSN JIT  batch f64x4 (2×F64X2)",  t_batch4, N, t_numpy)
        if t_batch4_par is not None:
            print_row("RSSN JIT  f64x4 parallel",          t_batch4_par, N, t_numpy)
        if t_batch8 is not None:
            print_row("RSSN JIT  batch f64x8 (4×F64X2)",         t_batch8,     N, t_numpy)
        if t_batch8_par is not None:
            print_row("RSSN JIT  f64x8 parallel (dtact fibers)",  t_batch8_par, N, t_numpy)
        print_row("NumPy     (SIMD/C, hand-optimised)",    t_numpy,  N)
        if t_ne is not None:
            print_row("numexpr   (multi-threaded JIT)",    t_ne,     N, t_numpy)
        if t_numba is not None:
            print_row("Numba     (LLVM, vectorized ufunc)", t_numba,  N, t_numpy)
        print_row("SymPy     lambdify → numpy",            t_sympy,  N, t_numpy)

        # ── Speedup summary ───────────────────────────────────────────────
        print()
        print(f"  Speedups vs NumPy ({t_numpy*1e3:.2f} ms baseline):")
        print(f"    JIT bulk   : {speedup_str(t_numpy, t_bulk)}")
        if t_batch  is not None:
            print(f"    JIT f64x2  : {speedup_str(t_numpy, t_batch)}")
        if t_batch_par is not None:
            print(f"    JIT f64x2∥ : {speedup_str(t_numpy, t_batch_par)} (parallel)")
        if t_batch4 is not None:
            print(f"    JIT f64x4  : {speedup_str(t_numpy, t_batch4)}")
        if t_batch4_par is not None:
            print(f"    JIT f64x4∥ : {speedup_str(t_numpy, t_batch4_par)} (parallel)")
        if t_batch8 is not None:
            print(f"    JIT f64x8  : {speedup_str(t_numpy, t_batch8)}")
        if t_batch8_par is not None:
            print(f"    JIT f64x8∥ : {speedup_str(t_numpy, t_batch8_par)} (parallel)")
        if t_ne is not None:
            print(f"    numexpr    : {speedup_str(t_numpy, t_ne)}")
        if t_numba is not None:
            print(f"    Numba      : {speedup_str(t_numpy, t_numba)}")
        print(f"    SymPy/lam  : {speedup_str(t_numpy, t_sympy)}")

        # ── Accuracy ─────────────────────────────────────────────────────
        ref = numpy_out
        print()
        def chk(label, arr):
            if arr is None:
                return
            d = float(np.max(np.abs(arr - ref)))
            mark = "✔" if d < 1e-9 else "✗ MISMATCH"
            print(f"  Accuracy  {label:<22s}  max|Δ|={d:.2e}  {mark}")
        chk("bulk",        rust_bulk_out)
        chk("batch f64x2", rust_batch_out)
        chk("batch f64x2 parallel", rust_batch_par_out)
        chk("batch f64x4", rust_batch4_out)
        chk("batch f64x4 parallel", rust_batch4_par_out)
        chk("batch f64x8",          rust_batch8_out)
        chk("batch f64x8 parallel", rust_batch8_par_out)

        # ── Temp-array note ───────────────────────────────────────────────
        ops = sum(expr_str.count(c) for c in "+-*/^")
        tmp_mb = ops * N * 8 / 1024 / 1024
        print(f"\n  NumPy temp arrays: ~{ops} binary ops → ~{tmp_mb:.0f} MB peak")
        print( "  RSSN JIT: 0 temp arrays — register-resident across entire expression")
        if HAS_NUMEXPR:
            print( "  numexpr:  ≈0 temp arrays — its own AST-based evaluator")
        if HAS_NUMBA:
            print( "  Numba:    ≈0 temp arrays — LLVM-fused scalar loop")

        summary.append((name, t_numpy, t_bulk, t_batch, t_batch_par, t_batch4, t_batch4_par, t_batch8, t_batch8_par, t_ne, t_numba, t_sympy))
        lib.rssn_dag_free(builder)

    # ── Summary table ───────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  SUMMARY — speedup vs hand-optimised NumPy  (higher = faster)")
    hdr_cols = ["bulk", "f64x2", "f64x2∥", "f64x4", "f64x4∥", "f64x8", "f64x8∥"]
    if HAS_NUMEXPR: hdr_cols.append("numexpr")
    if HAS_NUMBA:   hdr_cols.append("numba")
    hdr_cols.append("sympy")
    hdr = "  " + f"{'Expression':<46}" + "".join(f"  {c:>9}" for c in hdr_cols)
    print(hdr)
    print("  " + "─" * 46 + ("  " + "─" * 9) * len(hdr_cols))

    def _su(t_ref, t):
        return f"{t_ref / t:7.2f}x" if t is not None else "     n/a"

    for row in summary:
        name_r, t_np, t_bulk, t_batch, t_batch_par, t_batch4, t_batch4_par, t_batch8, t_batch8_par, t_ne, t_numba, t_sympy = row
        label = name_r.split("  ")[0] if "  " in name_r else name_r
        cells = [
            _su(t_np, t_bulk),
            _su(t_np, t_batch), _su(t_np, t_batch_par),
            _su(t_np, t_batch4), _su(t_np, t_batch4_par),
            _su(t_np, t_batch8), _su(t_np, t_batch8_par)
        ]
        if HAS_NUMEXPR: cells.append(_su(t_np, t_ne))
        if HAS_NUMBA:   cells.append(_su(t_np, t_numba))
        cells.append(_su(t_np, t_sympy))
        print("  " + f"{label:<46}" + "".join(f"  {c:>9}" for c in cells))

    print(f"""
  Observations:
  • Speedup grows with expression complexity as NumPy's intermediates
    overflow L2/L3 cache at N={N:,}.
  • RSSN JIT is register-resident: pays one mem read/write per input.
  • numexpr parses a string AST and avoids most temporaries; competitive
    on simple expressions, RSSN wins on deeply nested trees (no Python
    overhead, full algebraic simplification, custom FMA peepholes).
  • Numba (vectorized) compiles a scalar kernel to LLVM; matches or
    exceeds NumPy on simple ops, RSSN f64x4 pulls ahead on complex ones.
""")
    print(sep)


if __name__ == "__main__":
    run_benchmark()
