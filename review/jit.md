# Module Review: `jit`

## 1. Performance Issues (High Severity)

### 1.1 Ineffective Prefetching
In `emit_variable_load`, the compiler emits a prefetch hint for an address immediately before loading from it:
```rust
let _hint = emit_prefetch_hint(builder, addr);
builder.ins().load(types::F64, MemFlags::new(), addr, 0)
```
Prefetching is only beneficial when performed significantly in advance of the load to hide memory latency. Emitting it immediately before the load provides zero benefit and adds the overhead of the prefetch instruction itself.

### 1.2 Global Lock Contention in Linker
The `JitCompiler` uses an `Arc<Mutex<HashMap<u32, usize>>>` for custom function lookups. The `symbol_lookup_fn` closure locks this mutex during the linking phase of every compilation. In a multi-threaded environment where many expressions are being compiled in parallel, this will become a significant bottleneck.

### 1.3 Per-Node SSA Management Overhead
The iterative codegen uses `Vec<Value>` and `Vec<Frame>` with small initial capacities. For large expressions, these vectors will undergo numerous reallocations.

## 2. Correctness & Numerical Issues

### 2.1 Dangerous Identity Folding with `EPSILON`
The JIT primitives (`simplify_add`, `simplify_mul`) use `f64::EPSILON` to perform identity folding:
```rust
if lhs.abs() < f64::EPSILON { Some(rhs) }
```
This is **incorrect** for a symbolic engine. If a user provides a very small but non-zero value (e.g., `1e-20`), the engine will silently treat it as zero. Symbolic computation should typically only fold exact `0.0` or `1.0` unless a fuzzy matching mode is explicitly enabled. This can lead to significant precision loss.

### 2.2 Non-Recoverable Division by Zero
Runtime division by zero is handled via `TrapCode::unwrap_user(1)`. This triggers a machine-level trap (e.g., `SIGILL` or `SIGFPE` on Linux), which typically terminates the process. There is no mechanism in the current FFI/C-API to catch these traps and return an error code to the user, making the engine "brittle" in production environments.

## 3. Deviations from Plan

### 3.1 "Peephole Pass" Implementation
The plan mentions a "peephole pass over the per-node IR emission". While some folding exists in `emit_operator`, it is very limited and suffers from the numerical issues mentioned above. It doesn't handle more complex identities or algebraic simplifications that the plan implies.

## 4. Suggestions
- Remove the ineffective prefetch hints or implement a proper look-ahead prefetching strategy for expression evaluation.
- Change identity folding to use exact equality (`== 0.0`) to preserve numerical precision.
- Use an `AtomicPtr` array or a lock-free map for the custom function registry to avoid linker contention.
- Implement a signal handler or use Cranelift's trap handling features to convert machine traps into recoverable `Result::Err` values.
- Pre-allocate or reuse work buffers for the iterative codegen to reduce allocation pressure.
