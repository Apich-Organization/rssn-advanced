//! Example 05: Flat C-ABI & Lifecycle Simulation
//!
//! This example shows how non-Rust clients (C, C++, Python, Julia) interact with
//! the flat C-compatible FFI surface exported by RSSN-Advanced.
//!
//! It simulates raw FFI operations, capturing panic-free transitions, opaque
//! handle creations, algebraic simplification, and dynamic JIT execution.
//!
//! Run with: `cargo run --example 05_c_api_flat_lifecycle --features cranelift-jit,cranelift-frontend,cranelift-native,cranelift-codegen,cranelift-module`

use std::ffi::CString;
use std::os::raw::c_void;
use rssn_advanced::ffi::{
    rssn_dag_add, rssn_dag_compile, rssn_dag_constant, rssn_dag_execute, rssn_dag_free,
    rssn_dag_new, rssn_dag_simplify, rssn_dag_variable, RssnStatus,
};

fn main() {
    println!("=== RSSN-Advanced Example 05: Flat C-ABI FFI Lifecycle ===\n");

    // 1. Simulating C client: allocating an opaque builder handle
    println!("Simulating FFI: rssn_dag_new()...");
    let builder_ptr = rssn_dag_new();
    assert!(!builder_ptr.is_null(), "FFI Builder pointer is NULL!");
    println!("  Opaque builder context successfully allocated at address: {:?}\n", builder_ptr);

    // 2. Simulating C client: constructing variable and constant nodes
    println!("Simulating FFI: rssn_dag_variable() & rssn_dag_constant()...");
    let x_name = CString::new("x").unwrap();
    let y_name = CString::new("y").unwrap();

    let id_x = rssn_dag_variable(builder_ptr, x_name.as_ptr());
    let id_y = rssn_dag_variable(builder_ptr, y_name.as_ptr());
    let id_c = rssn_dag_constant(builder_ptr, 10.0);

    println!("  Allocated Variable 'x' with index  : {}", id_x);
    println!("  Allocated Variable 'y' with index  : {}", id_y);
    println!("  Allocated Constant '10.0' index    : {}\n", id_c);

    // 3. Simulating C client: constructing operations: (x + y) + 10.0
    println!("Simulating FFI: rssn_dag_add()...");
    let add1 = rssn_dag_add(builder_ptr, id_x, id_y);
    let root_expr = rssn_dag_add(builder_ptr, add1, id_c);
    println!("  Allocated addition operation root  : {}\n", root_expr);

    // 4. Simulating C client: algebraic simplification
    println!("Simulating FFI: rssn_dag_simplify()...");
    let simplified_expr = rssn_dag_simplify(builder_ptr, root_expr);
    println!("  Simplified root node index returned: {}\n", simplified_expr);

    // 5. Simulating C client: compiling to native dynamic JIT function
    println!("Simulating FFI: rssn_dag_compile()...");
    let mut jit_function_ptr: *mut c_void = std::ptr::null_mut();
    
    // Pass pointer to JIT function pointer to write back address
    let status = rssn_dag_compile(builder_ptr, simplified_expr, &mut jit_function_ptr);

    if status == RssnStatus::Success {
        println!("  JIT Compilation Successful!");
        println!("  Function pointer generated at address: {:?}\n", jit_function_ptr);

        // 6. Simulating C client: executing native dynamic JIT function
        // Input float buffer in order of variables x=0, y=1: x = 5.0, y = 3.0
        let inputs = vec![5.0, 3.0];
        println!("Simulating FFI: rssn_dag_execute() with inputs x=5.0, y=3.0...");
        
        let result = rssn_dag_execute(jit_function_ptr, inputs.as_ptr());
        println!("  Native evaluation completed!");
        println!("  Calculated Result                  : {:.6}", result);
        println!("  Expected mathematical sum          : 18.000000\n");
    } else {
        println!("  [!] JIT Compilation is skipped or not compiled with features. Status code: {:?}", status);
        println!("      Ensure 'cranelift-jit' cargo feature is enabled to compile dynamically.\n");
    }

    // 7. Simulating C client: releasing opaque context builder memory
    println!("Simulating FFI: rssn_dag_free()...");
    rssn_dag_free(builder_ptr);
    println!("  Opaque builder memory successfully deallocated & reclaimed.");

    println!("===========================================================");
}
