//! SIMD-optimized preset function library.
//!
//! Provides hardware-accelerated batch operations as the default
//! high-performance execution path. Falls back to scalar implementations
//! when SIMD is unavailable.
//!
//! - `arithmetic` — Batch coefficient add, mul, and comparison.
//! - `hash` — Vectorized rapidhash for batch node deduplication.
//! - `detect` — Runtime CPU feature detection (SSE4.2, AVX2, NEON).

pub mod arithmetic;
pub mod detect;
pub mod hash;

pub use arithmetic::{batch_add, batch_add_scalar, batch_mul};
pub use detect::{has_avx2, has_sse42};
pub use hash::batch_hash;
