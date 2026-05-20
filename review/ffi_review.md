# RSSN-Advanced Review: `src/ffi`

## **1. Alignment with `plan.md` & Instructions**

### **Async-Fiber FFI / `dtact`**
- **Status**: **CRITICAL FAIL**
- **Issue**: The user explicitly instructed: "the async-fiber ffi shall use dtact".
- **Evidence**: `src/ffi/async_bridge.rs` uses `std::thread::spawn` to perform asynchronous work. This is OS-thread based, not fiber-based, and does not use the `dtact` crate as required.
- **Impact**: Spawning a full OS thread for every async simplification call is extremely heavy and contradicts the "high-performance" and "industrial-grade stability" goals of the project.

### **C/C++ Foreign Function Interface**
- **Status**: **PASS (Partial)**
- **Observation**: Uses flat C API with opaque handles and `catch_unwind`.
- **Issue**: `rssn_dag_compile` returns `RssnStatus`, but `rssn_dag_variable`, `rssn_dag_add`, etc., return `u32` (node ID) and use `u32::MAX` for errors. This inconsistency makes error handling on the C side messy.

---

## **2. Performance Issues**

### **Thread Spawning Overhead**
- **Issue**: `rssn_dag_simplify_async` spawns a new thread per request.
- **Recommendation**: Integrate `dtact` as requested to use lightweight fibers or a task-based executor.

### **String Allocation in Hot Path**
- **Issue**: `rssn_dag_variable` uses `c_str.to_string_lossy()`.
- **Observation**: This may allocate a new `String` for every variable lookup.
- **Recommendation**: Use `CStr` directly to look up names in the symbol registry to avoid allocations.

---

## **3. Safety & Security**

### **Panic Safety**
- **Status**: **PASS**
- **Observation**: `catch_unwind` is used at all FFI boundaries, preventing UB from Rust panics crossing into C.

### **Pointer Validation**
- **Issue**: While `builder.is_null()` is checked, there is no validation that the `builder` pointer actually points to a valid `DagBuilder` instance (e.g., no "magic number" or tracking). This is standard for C APIs but worth noting.

---

## **4. Error Handling**

### **Macro Non-Compliance**
- **Issue**: The FFI layer does not use the requested cold-path error macro.
- **Recommendation**: Errors in the FFI should be handled via the macro to ensure they are optimized as cold paths.

---

## **5. Extensibility**

### **Flat API Surface**
- **Observation**: The API is very flat and functional. Adding new features (like SIMD batching via FFI) will require adding many new `rssn_dag_...` functions.
- **Recommendation**: Consider a "request/response" style FFI for complex operations if the number of functions becomes unmanageable, or stick to the current flat style but organize by submodule.
