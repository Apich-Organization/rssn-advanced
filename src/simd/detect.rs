//! Runtime CPU feature detection.
//!
//! Detects available SIMD instruction sets at runtime
//! (`SSE4.2`, `AVX2` on `x86_64`; NEON on `AArch64`) and selects the
//! optimal path.

/// Detects if AVX2 is available on the host CPU.
#[must_use]
pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Detects if SSE4.2 is available on the host CPU.
#[must_use]
pub fn has_sse42() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("sse4.2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
