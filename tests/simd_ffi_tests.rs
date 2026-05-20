//! Integration tests for SIMD vectorization and Cross-Language FFI.

#[cfg(test)]
mod simd_ffi_tests {
    use std::ffi::CString;
    use std::os::raw::c_void;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use rssn_advanced::simd::{batch_add, batch_add_scalar, batch_hash, batch_mul, has_avx2};
    use rssn_advanced::ffi::{
        rssn_dag_add, rssn_dag_compile, rssn_dag_constant, rssn_dag_execute, rssn_dag_free,
        rssn_dag_new, rssn_dag_simplify, rssn_dag_simplify_async, rssn_dag_variable, RssnStatus,
    };

    #[test]
    fn test_simd_arithmetic_and_hashing() {
        // Test CPU detection
        let _ = has_avx2();

        // Test batch_add
        let lhs = vec![1.0, 2.0, 3.0, 4.0];
        let rhs = vec![10.0, 20.0, 30.0, 40.0];
        let mut result = vec![0.0; 4];
        batch_add(&lhs, &rhs, &mut result).expect("batch_add");
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);

        // Test batch_mul
        let mut result_mul = vec![0.0; 4];
        batch_mul(&lhs, &rhs, &mut result_mul).expect("batch_mul");
        assert_eq!(result_mul, vec![10.0, 40.0, 90.0, 160.0]);

        // Test batch_add_scalar
        let mut result_scalar = vec![0.0; 4];
        batch_add_scalar(&lhs, 5.0, &mut result_scalar).expect("batch_add_scalar");
        assert_eq!(result_scalar, vec![6.0, 7.0, 8.0, 9.0]);

        // Test batch_hash
        let keys = vec![100, 200, 300, 400];
        let mut hashes = vec![0; 4];
        batch_hash(&keys, &mut hashes).expect("batch_hash");
        assert_ne!(hashes[0], hashes[1]);
        assert_ne!(hashes[2], hashes[3]);
    }

    #[test]
    fn test_ffi_builder_and_jit_lifecycle() {
        let builder = rssn_dag_new();
        assert!(!builder.is_null());

        let x_name = CString::new("x").unwrap();
        let y_name = CString::new("y").unwrap();

        let x = rssn_dag_variable(builder, x_name.as_ptr());
        let y = rssn_dag_variable(builder, y_name.as_ptr());
        let c = rssn_dag_constant(builder, 10.0);

        let add1 = rssn_dag_add(builder, x, y);
        let add2 = rssn_dag_add(builder, add1, c);

        assert_ne!(add2, u32::MAX);

        // Simplify expression
        let simplified = rssn_dag_simplify(builder, add2);
        assert_ne!(simplified, u32::MAX);

        // Compile expression (requires cranelift JIT feature)
        let mut func_ptr: *mut c_void = std::ptr::null_mut();
        let status = rssn_dag_compile(builder, simplified, &mut func_ptr);

        if status == RssnStatus::Success {
            assert!(!func_ptr.is_null());
            // Execute: x = 5.0, y = 3.0, (5.0 + 3.0) + 10.0 = 18.0
            let variables = vec![5.0, 3.0];
            let val = rssn_dag_execute(func_ptr, variables.as_ptr());
            assert_eq!(val, 18.0);
        }

        rssn_dag_free(builder);
    }

    struct AsyncState {
        simplified_root: u32,
        status: RssnStatus,
        fired: bool,
    }

    unsafe extern "C" fn ffi_callback(
        simplified_root: u32,
        status: RssnStatus,
        user_data: *mut c_void,
    ) {
        let state_lock = unsafe { &*(user_data as *const Mutex<AsyncState>) };
        let mut state = state_lock.lock().unwrap();
        state.simplified_root = simplified_root;
        state.status = status;
        state.fired = true;
    }

    #[test]
    fn test_ffi_async_simplification() {
        let builder = rssn_dag_new();
        let x_name = CString::new("x").unwrap();
        let x = rssn_dag_variable(builder, x_name.as_ptr());
        let y = rssn_dag_constant(builder, 4.0);
        let expr = rssn_dag_add(builder, x, y);

        let state = Arc::new(Mutex::new(AsyncState {
            simplified_root: u32::MAX,
            status: RssnStatus::NullPointer,
            fired: false,
        }));

        let raw_state = Arc::as_ptr(&state) as *mut c_void;

        rssn_dag_simplify_async(builder, expr, ffi_callback, raw_state);

        // Wait up to 500ms for thread completion
        let start = std::time::Instant::now();
        while start.elapsed() < Duration::from_millis(500) {
            {
                let lock = state.lock().unwrap();
                if lock.fired {
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(10));
        }

        let lock = state.lock().unwrap();
        assert!(lock.fired);
        assert_eq!(lock.status, RssnStatus::Success);
        assert_ne!(lock.simplified_root, u32::MAX);

        rssn_dag_free(builder);
    }
}
