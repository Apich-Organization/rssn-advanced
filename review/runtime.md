# Module Review: `runtime`

## 1. Performance Issues (High Severity)

### 1.1 Mutex Contention in Parallel Loop
The `parallel_for_each` utility, which is the heart of the parallel solver and async FFI, uses a single `Arc<Mutex<Vec<Option<T>>>>` to collect results from all fibers:
```rust
handles.push(spawn_task(gate, move || {
    let value = task();
    if let Ok(mut guard) = slots_arc.lock() {
        *slot = Some(value);
    }
}));
```
Every worker fiber must acquire a global mutex lock simply to write its result into a pre-assigned, unique slot. This introduces a major serialization point in what is supposed to be a parallel operation, likely negating the performance benefits of using fibers for small tasks.

### 1.2 Excessive Allocations per Task Spawn
`spawn_task` performs two separate heap allocations for every task spawn due to "double-boxing" for C FFI compatibility:
```rust
let boxed: Box<dyn FnOnce() + Send + 'static> = Box::new(f);
let arg: *mut Box<dyn FnOnce() + Send + 'static> = Box::into_raw(Box::new(boxed));
```
When combined with the allocations in the parallel solver and the JIT, this adds to the overall allocation pressure that plagues the entire engine.

## 2. Engineering Standards

### 2.1 Lack of Panic Propagation in `parallel_for_each`
If a fiber panics, the `parallel_for_each` function simply flattens the result vector (`drained.into_iter().flatten()`), silently ignoring any tasks that failed. This can lead to incorrect results where partial sums are returned instead of an error, violating the "correctness" requirement.

## 3. Suggestions
- Use a lock-free approach for collecting results in `parallel_for_each`, such as writing to a raw slice with atomic synchronization or using a specialized concurrent collection.
- Optimize the trampoline to require only a single allocation for the task closure.
- Ensure panics in worker fibers are captured and propagated to the caller as an error or by re-panicking the main thread.
