//! Generic, allocator-light utilities shared across modules.
//!
//! Nothing here is symbolic-math-aware. Code that lives here must compile
//! without pulling in `dag`, `ast`, `jit`, etc.

pub mod worklist;
