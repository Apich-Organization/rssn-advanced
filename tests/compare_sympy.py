#!/usr/bin/env python3
"""
Performance comparison script between SymPy and RSSN-Advanced JIT.

Loads the compiled Rust shared library via ctypes, constructs the expression:
    (x + y) + 10.0
And benchmarks evaluation against SymPy's subs() and lambdify() over 1,000,000 iterations.
"""

import ctypes
import os
import time
import random
import sympy

# 1. Load the shared library
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../target/release/librssn_advanced.so"))
if not os.path.exists(lib_path):
    raise FileNotFoundError(f"Shared library not found at {lib_path}. Please run 'cargo build --release' first.")

lib = ctypes.CDLL(lib_path)

# 2. Define FFI signatures
lib.rssn_dag_new.argtypes = []
lib.rssn_dag_new.restype = ctypes.c_void_p

lib.rssn_dag_free.argtypes = [ctypes.c_void_p]
lib.rssn_dag_free.restype = None

lib.rssn_dag_variable.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.rssn_dag_variable.restype = ctypes.c_uint32

lib.rssn_dag_constant.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.rssn_dag_constant.restype = ctypes.c_uint32

lib.rssn_dag_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
lib.rssn_dag_add.restype = ctypes.c_uint32

lib.rssn_dag_simplify.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
lib.rssn_dag_simplify.restype = ctypes.c_uint32

lib.rssn_dag_compile.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
lib.rssn_dag_compile.restype = ctypes.c_int  # Returns RssnStatus

lib.rssn_dag_execute.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
lib.rssn_dag_execute.restype = ctypes.c_double


def run_benchmark():
    print("=" * 60)
    print("       RSSN-ADVANCED JIT vs SYMPY BENCHMARK VALIDATION")
    print("=" * 60)
    
    # --- Rust JIT Setup ---
    builder = lib.rssn_dag_new()
    assert builder, "Failed to create DagBuilder context."

    # Construct expression: (x + y) + 10.0
    x_id = lib.rssn_dag_variable(builder, b"x")
    y_id = lib.rssn_dag_variable(builder, b"y")
    const_id = lib.rssn_dag_constant(builder, ctypes.c_double(10.0))

    add1 = lib.rssn_dag_add(builder, x_id, y_id)
    expr_id = lib.rssn_dag_add(builder, add1, const_id)

    # Simplify
    simplified_id = lib.rssn_dag_simplify(builder, expr_id)

    # Compile
    func_ptr = ctypes.c_void_p()
    status = lib.rssn_dag_compile(builder, simplified_id, ctypes.byref(func_ptr))
    assert status == 0, f"Compilation failed with status code {status}."
    assert func_ptr.value, "Returned JIT function pointer is NULL."

    print("[✔] Rust JIT compilation successful.")

    # --- SymPy Setup ---
    x, y = sympy.symbols('x y')
    sympy_expr = (x + y) + 10.0
    
    # Create standard substitutions and lambdify version (SymPy's optimized path)
    sympy_lambdified = sympy.lambdify((x, y), sympy_expr, 'math')

    print("[✔] SymPy setup and lambdify compilation successful.")

    # --- Generate Random Test Inputs ---
    iterations = 500000
    print(f"\nGenerating {iterations} random input pairs...")
    inputs = [(random.uniform(-100.0, 100.0), random.uniform(-100.0, 100.0)) for _ in range(iterations)]

    # --- Benchmark Rust JIT ---
    print("\nBenchmarking RSSN-Advanced JIT...")
    start_time = time.perf_counter()
    
    # Reuse a buffer for the ctypes array to avoid allocation overhead inside the loop
    double_array_t = ctypes.c_double * 2
    val_buffer = double_array_t()
    
    rust_results = []
    for val_x, val_y in inputs:
        val_buffer[0] = val_x
        val_buffer[1] = val_y
        res = lib.rssn_dag_execute(func_ptr, val_buffer)
        rust_results.append(res)
        
    rust_duration = time.perf_counter() - start_time
    avg_rust = (rust_duration / iterations) * 1e9

    print(f"  Total Time : {rust_duration:.4f} seconds")
    print(f"  Average    : {avg_rust:.2f} ns per evaluation")

    # --- Benchmark SymPy Lambdify ---
    print("\nBenchmarking SymPy Lambdify...")
    start_time = time.perf_counter()
    sympy_results = []
    for val_x, val_y in inputs:
        res = sympy_lambdified(val_x, val_y)
        sympy_results.append(res)
        
    sympy_duration = time.perf_counter() - start_time
    avg_sympy = (sympy_duration / iterations) * 1e9

    print(f"  Total Time : {sympy_duration:.4f} seconds")
    print(f"  Average    : {avg_sympy:.2f} ns per evaluation")

    # --- Benchmark SymPy Standard Subs (on a subset to avoid extreme slowness) ---
    print("\nBenchmarking SymPy Standard subs() (on 1000 items)...")
    start_time = time.perf_counter()
    for val_x, val_y in inputs[:1000]:
        _ = float(sympy_expr.subs({x: val_x, y: val_y}))
    subs_duration = time.perf_counter() - start_time
    avg_subs = (subs_duration / 1000) * 1e9
    print(f"  Average    : {avg_subs:.2f} ns per evaluation")

    # --- Accuracy Verification ---
    print("\nVerifying numerical accuracy...")
    mismatches = 0
    for idx, (rx, sy) in enumerate(zip(rust_results, sympy_results)):
        if abs(rx - sy) > 1e-12:
            mismatches += 1
            if mismatches < 5:
                print(f"  Mismatch at index {idx}: Rust={rx}, SymPy={sy}")
                
    if mismatches == 0:
        print("  [✔] 100% accurate! Rust results match SymPy exactly.")
    else:
        print(f"  [✘] Found {mismatches} accuracy mismatches.")

    # --- Speedup Report ---
    speedup_lambdify = avg_sympy / avg_rust
    speedup_subs = avg_subs / avg_rust
    
    print("\n" + "=" * 60)
    print("                   PERFORMANCE SPEEDUP SUMMARY")
    print("=" * 60)
    print(f"  RSSN-Advanced vs SymPy Lambdify : {speedup_lambdify:.2f}x FASTER")
    print(f"  RSSN-Advanced vs SymPy subs()   : {speedup_subs:.2f}x FASTER")
    print("=" * 60)

    # Clean up builder context
    lib.rssn_dag_free(builder)


if __name__ == "__main__":
    run_benchmark()
