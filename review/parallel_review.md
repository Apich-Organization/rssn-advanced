# RSSN-Advanced Review: `src/parallel`

## **1. Alignment with `plan.md`**

### **Commutativity & Permission Control**
- **Status**: **PASS**
- **Observation**: `SymbolPermissions` and `splitter.rs` correctly implement the plan's strategy for identifying parallelizable chunks based on algebraic properties and user-defined permissions.

### **Avoidance of MESI Bus Synchronization**
- **Status**: **PASS**
- **Observation**: `ThreadLocalState` correctly uses `#[repr(align(128))]` to prevent false sharing, which is a key requirement in §4.2 of the plan. It also uses `Acquire`/`Release` ordering as requested.

### **Staged Simplification**
- **Status**: **FAIL**
- **Issue**: While `SimplifyConfig` exists, there is no implementation of the "stage-wise trigger" or "intermediate rounds" of simplification mentioned in §4.1. The logic is missing.

---

## **2. Performance Issues**

### **Massive Arena Cloning**
- **Issue**: `parallel_evaluate` performs `arena_clone = arena.clone()` for every chunk.
- **Critical Impact**: If a DAG has 1 million nodes and is split into 16 chunks, the system will allocate and copy 16 million nodes. This completely defeats the "Global DAG" shared storage model and will likely cause an OOM (Out of Memory) or severe performance degradation.
- **Recommendation**: Pass the arena as an `Arc<DagArena>` or use a read-only reference across threads. The nodes in the arena should be immutable during evaluation.

### **Thread Spawning Overhead**
- **Issue**: `parallel_evaluate` uses `thread::spawn` per chunk.
- **Recommendation**: As with the FFI module, this should use a thread pool or the requested `dtact` fiber system to avoid OS-level thread creation overhead.

### **Recursive Evaluation**
- **Issue**: `evaluate_node` is recursive.
- **Risk**: Stack overflow on deep expressions.
- **Recommendation**: Use an iterative evaluator with an explicit stack.

---

## **3. Zero-Copy & `bincode-next`**

### **Arena Copying**
- **Issue**: The frequent cloning of `DagArena` is the antithesis of the "zero-copy" goal. Even if serialization is zero-copy, the runtime behavior is highly wasteful.

---

## **4. Error Handling**

### **Panic Risk**
- **Issue**: `evaluate_node` uses `.unwrap_or(0.0)` for variables and operators.
- **Observation**: This "graceful" failure masks potential bugs (e.g., missing variable values). It should use the requested cold-path error macro to report missing data.

---

## **5. Extensibility**

### **Closed Operator Set**
- **Issue**: `evaluate_node` has a hardcoded `match op` for `OpKind`.
- **Impact**: Adding new operators requires modifying this core evaluation loop.
