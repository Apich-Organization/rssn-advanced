//! Runtime CPU feature detection.
//!
//! Detects available SIMD instruction sets at runtime
//! (`SSE4.2`, `AVX2` on `x86_64`; NEON on `AArch64`) and selects the
//! optimal path.

/// Detects if AVX2 is available on the host CPU.
#[must_use]
#[allow(clippy::missing_const_for_fn)]
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
#[allow(clippy::missing_const_for_fn)]
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

/// Detects if NEON is available on the host CPU.
///
/// On aarch64, NEON is mandatory per ARMv8-A spec, so this always returns
/// `true` on that target. On other architectures it returns `false`.
#[must_use]
#[allow(clippy::missing_const_for_fn)]
pub fn has_neon() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}
