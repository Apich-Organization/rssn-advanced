# Module Review: `dag` (Phase 3 Audit)

## 1. Architectural Integrity

### 1.1 The 80-Byte Albatross
The core `DagNode` remains 80 bytes. 
- **Sharp Question:** If the goal is "industrial-scale" symbolic computation with millions of nodes, why are we settling for a data structure that is 2.5x larger than our own 32-byte wire format? Are we prioritizing the convenience of `Option<f64>` and `Vec<DagNodeId>` over the massive L1/L2 cache pressure they create?

### 1.2 Redundant Numeric Storage
`DagNode` carries both an `Option<f64>` and `NodeMetadata` with a `coefficient: f64`.
- **Sharp Question:** Why do we have two separate fields for numeric values? Could we not represent constants as nodes where the `coefficient` *is* the value, or is this dual-tracking of numbers a signal that our metadata model is poorly defined?

## 2. Correctness & Performance

### 2.1 Bucket-Scan Bottleneck
`DedupMap` still uses `HashMap<u64, Vec<DagNodeId>>`. 
- **Sharp Question:** In a "perfectly shared" DAG, collisions are common for similar structures. Why are we performing a linear scan of a `Vec` for every `get_or_insert` instead of using a flat, open-addressed hash table? Is our "RapidHash" gains being squandered on `std::collections::HashMap` overhead?

## 3. Extensibility

### 3.1 Hardcoded Operator Universe
`OpKind` is a closed enum. 
- **Sharp Question:** How does a user implement a "Softmax" operator or a "Quantum Gate" without modifying the core library? Is this a "pluggable symbolic core" or a fixed mathematical calculator?

## 4. Dead Code

### 4.1 Orphaned Flags
`NodeFlags::CANONICAL` is defined but ignored by the `HeuristicEngine`, which uses a local `HashSet`.
- **Sharp Question:** Why define a persistent bit in the metadata if the primary consumer (the engine) is just going to allocate a transient `HashSet` on every call anyway?
