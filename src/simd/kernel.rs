//! Runtime-dispatched SIMD kernel abstraction.
//!
//! [`SimdKernel`] is a trait that abstracts batch floating-point arithmetic.
//! Concrete implementations cover the scalar fallback, AVX2, and NEON paths.
//!
//! [`global_kernel`] selects the best available kernel once per process via
//! `OnceLock` — subsequent calls are a single pointer load.

use std::sync::OnceLock;

// =========================================================================
// Trait
// =========================================================================

/// Batch arithmetic operations, abstracted over SIMD width and ISA.
///
/// Implementations must be `Send + Sync` so the global singleton is safe
/// across thread boundaries. Each method operates element-wise over equal-
/// length slices; callers guarantee slice lengths match.
pub trait SimdKernel: Send + Sync {
    /// Element-wise addition: `out[i] = a[i] + b[i]`.
    fn batch_add(&self, a: &[f64], b: &[f64], out: &mut [f64]);
    /// Element-wise subtraction: `out[i] = a[i] - b[i]`.
    fn batch_sub(&self, a: &[f64], b: &[f64], out: &mut [f64]);
    /// Element-wise multiplication: `out[i] = a[i] * b[i]`.
    fn batch_mul(&self, a: &[f64], b: &[f64], out: &mut [f64]);
    /// Element-wise division: `out[i] = a[i] / b[i]`.
    fn batch_div(&self, a: &[f64], b: &[f64], out: &mut [f64]);
    /// Fused multiply-add: `out[i] = a[i] * b[i] + c[i]`.
    fn batch_fma(&self, a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]);
    /// Element-wise square root: `out[i] = sqrt(a[i])`.
    fn batch_sqrt(&self, a: &[f64], out: &mut [f64]);
    /// Element-wise negation: `out[i] = -a[i]`.
    fn batch_neg(&self, a: &[f64], out: &mut [f64]);
    /// Element-wise absolute value: `out[i] = |a[i]|`.
    fn batch_abs(&self, a: &[f64], out: &mut [f64]);
    /// Short name for logging/introspection.
    fn name(&self) -> &'static str;
}

// =========================================================================
// ScalarKernel — pure Rust fallback, always available
// =========================================================================

/// Pure-scalar fallback kernel. Works on every platform.
pub struct ScalarKernel;

impl SimdKernel for ScalarKernel {
    fn batch_add(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        for i in 0..out.len() {
            out[i] = a[i] + b[i];
        }
    }

    fn batch_sub(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        for i in 0..out.len() {
            out[i] = a[i] - b[i];
        }
    }

    fn batch_mul(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        for i in 0..out.len() {
            out[i] = a[i] * b[i];
        }
    }

    fn batch_div(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        for i in 0..out.len() {
            out[i] = a[i] / b[i];
        }
    }

    fn batch_fma(&self, a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
        for i in 0..out.len() {
            out[i] = a[i].mul_add(b[i], c[i]);
        }
    }

    fn batch_sqrt(&self, a: &[f64], out: &mut [f64]) {
        for i in 0..out.len() {
            out[i] = a[i].sqrt();
        }
    }

    fn batch_neg(&self, a: &[f64], out: &mut [f64]) {
        for i in 0..out.len() {
            out[i] = -a[i];
        }
    }

    fn batch_abs(&self, a: &[f64], out: &mut [f64]) {
        for i in 0..out.len() {
            out[i] = a[i].abs();
        }
    }

    fn name(&self) -> &'static str {
        "scalar"
    }
}

// =========================================================================
// Avx2Kernel — x86_64 AVX2 + FMA path
// =========================================================================

/// AVX2/FMA-accelerated kernel.
///
/// Guarded at construction time by a runtime `has_avx2()` check;
/// never instantiated on non-x86_64 targets.
#[cfg(target_arch = "x86_64")]
pub struct Avx2Kernel;

#[cfg(target_arch = "x86_64")]
impl SimdKernel for Avx2Kernel {
    fn batch_add(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        use crate::asm_presets::add_f64x4;
        const LANES: usize = 4;
        let n = out.len();
        let mut i = 0;
        while i + LANES <= n {
            add_f64x4::apply(&a[i..i + LANES], &b[i..i + LANES], &mut out[i..i + LANES]);
            i += LANES;
        }
        while i < n {
            out[i] = a[i] + b[i];
            i += 1;
        }
    }

    fn batch_sub(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        use crate::asm_presets::sub_f64x4;
        const LANES: usize = 4;
        let n = out.len();
        let mut i = 0;
        while i + LANES <= n {
            sub_f64x4::apply(&a[i..i + LANES], &b[i..i + LANES], &mut out[i..i + LANES]);
            i += LANES;
        }
        while i < n {
            out[i] = a[i] - b[i];
            i += 1;
        }
    }

    fn batch_mul(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        use crate::asm_presets::mul_f64x4;
        const LANES: usize = 4;
        let n = out.len();
        let mut i = 0;
        while i + LANES <= n {
            mul_f64x4::apply(&a[i..i + LANES], &b[i..i + LANES], &mut out[i..i + LANES]);
            i += LANES;
        }
        while i < n {
            out[i] = a[i] * b[i];
            i += 1;
        }
    }

    fn batch_div(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        use crate::asm_presets::div_f64x4;
        const LANES: usize = 4;
        let n = out.len();
        let mut i = 0;
        while i + LANES <= n {
            div_f64x4::apply(&a[i..i + LANES], &b[i..i + LANES], &mut out[i..i + LANES]);
            i += LANES;
        }
        while i < n {
            out[i] = a[i] / b[i];
            i += 1;
        }
    }

    fn batch_fma(&self, a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
        use crate::asm_presets::fma_f64x4;
        const LANES: usize = 4;
        let len = out.len();
        let mut idx = 0;
        while idx + LANES <= len {
            fma_f64x4::apply(
                &a[idx..idx + LANES],
                &b[idx..idx + LANES],
                &c[idx..idx + LANES],
                &mut out[idx..idx + LANES],
            );
            idx += LANES;
        }
        while idx < len {
            out[idx] = a[idx].mul_add(b[idx], c[idx]);
            idx += 1;
        }
    }

    fn batch_sqrt(&self, a: &[f64], out: &mut [f64]) {
        use crate::asm_presets::sqrt_f64x4;
        const LANES: usize = 4;
        let n = out.len();
        let mut i = 0;
        while i + LANES <= n {
            sqrt_f64x4::apply(&a[i..i + LANES], &mut out[i..i + LANES]);
            i += LANES;
        }
        while i < n {
            out[i] = a[i].sqrt();
            i += 1;
        }
    }

    fn batch_neg(&self, a: &[f64], out: &mut [f64]) {
        use crate::asm_presets::neg_f64x4;
        const LANES: usize = 4;
        let n = out.len();
        let mut i = 0;
        while i + LANES <= n {
            neg_f64x4::apply(&a[i..i + LANES], &mut out[i..i + LANES]);
            i += LANES;
        }
        while i < n {
            out[i] = -a[i];
            i += 1;
        }
    }

    fn batch_abs(&self, a: &[f64], out: &mut [f64]) {
        use crate::asm_presets::abs_f64x4;
        const LANES: usize = 4;
        let n = out.len();
        let mut i = 0;
        while i + LANES <= n {
            abs_f64x4::apply(&a[i..i + LANES], &mut out[i..i + LANES]);
            i += LANES;
        }
        while i < n {
            out[i] = a[i].abs();
            i += 1;
        }
    }

    fn name(&self) -> &'static str {
        "avx2"
    }
}

// =========================================================================
// NeonKernel — aarch64 NEON path
// =========================================================================

/// NEON-accelerated kernel for AArch64.
#[cfg(target_arch = "aarch64")]
pub struct NeonKernel;

#[cfg(target_arch = "aarch64")]
impl SimdKernel for NeonKernel {
    fn batch_add(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        use crate::asm_presets::add_f64x2;
        const LANES: usize = 2;
        let n = out.len();
        let mut i = 0;
        while i + LANES <= n {
            let a2: &[f64; 2] = (&a[i..i + LANES]).try_into().unwrap();
            let b2: &[f64; 2] = (&b[i..i + LANES]).try_into().unwrap();
            let o2: &mut [f64; 2] = (&mut out[i..i + LANES]).try_into().unwrap();
            add_f64x2::apply(a2, b2, o2);
            i += LANES;
        }
        while i < n {
            out[i] = a[i] + b[i];
            i += 1;
        }
    }

    fn batch_sub(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        use crate::asm_presets::sub_f64x2;
        const LANES: usize = 2;
        let n = out.len();
        let mut i = 0;
        while i + LANES <= n {
            let a2: &[f64; 2] = (&a[i..i + LANES]).try_into().unwrap();
            let b2: &[f64; 2] = (&b[i..i + LANES]).try_into().unwrap();
            let o2: &mut [f64; 2] = (&mut out[i..i + LANES]).try_into().unwrap();
            sub_f64x2::apply(a2, b2, o2);
            i += LANES;
        }
        while i < n {
            out[i] = a[i] - b[i];
            i += 1;
        }
    }

    fn batch_mul(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        use crate::asm_presets::mul_f64x2;
        const LANES: usize = 2;
        let n = out.len();
        let mut i = 0;
        while i + LANES <= n {
            let a2: &[f64; 2] = (&a[i..i + LANES]).try_into().unwrap();
            let b2: &[f64; 2] = (&b[i..i + LANES]).try_into().unwrap();
            let o2: &mut [f64; 2] = (&mut out[i..i + LANES]).try_into().unwrap();
            mul_f64x2::apply(a2, b2, o2);
            i += LANES;
        }
        while i < n {
            out[i] = a[i] * b[i];
            i += 1;
        }
    }

    fn batch_div(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        use crate::asm_presets::div_f64x2;
        const LANES: usize = 2;
        let n = out.len();
        let mut i = 0;
        while i + LANES <= n {
            let a2: &[f64; 2] = (&a[i..i + LANES]).try_into().unwrap();
            let b2: &[f64; 2] = (&b[i..i + LANES]).try_into().unwrap();
            let o2: &mut [f64; 2] = (&mut out[i..i + LANES]).try_into().unwrap();
            div_f64x2::apply(a2, b2, o2);
            i += LANES;
        }
        while i < n {
            out[i] = a[i] / b[i];
            i += 1;
        }
    }

    fn batch_fma(&self, a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
        // NEON FMA falls back to scalar mul_add (vfma intrinsic not exposed for f64x2 in stable).
        for i in 0..out.len() {
            out[i] = a[i].mul_add(b[i], c[i]);
        }
    }

    fn batch_sqrt(&self, a: &[f64], out: &mut [f64]) {
        use crate::asm_presets::sqrt_f64x2;
        const LANES: usize = 2;
        let n = out.len();
        let mut i = 0;
        while i + LANES <= n {
            let a2: &[f64; 2] = (&a[i..i + LANES]).try_into().unwrap();
            let o2: &mut [f64; 2] = (&mut out[i..i + LANES]).try_into().unwrap();
            sqrt_f64x2::apply(a2, o2);
            i += LANES;
        }
        while i < n {
            out[i] = a[i].sqrt();
            i += 1;
        }
    }

    fn batch_neg(&self, a: &[f64], out: &mut [f64]) {
        use crate::asm_presets::neg_f64x2;
        const LANES: usize = 2;
        let n = out.len();
        let mut i = 0;
        while i + LANES <= n {
            let a2: &[f64; 2] = (&a[i..i + LANES]).try_into().unwrap();
            let o2: &mut [f64; 2] = (&mut out[i..i + LANES]).try_into().unwrap();
            neg_f64x2::apply(a2, o2);
            i += LANES;
        }
        while i < n {
            out[i] = -a[i];
            i += 1;
        }
    }

    fn batch_abs(&self, a: &[f64], out: &mut [f64]) {
        use crate::asm_presets::abs_f64x2;
        const LANES: usize = 2;
        let n = out.len();
        let mut i = 0;
        while i + LANES <= n {
            let a2: &[f64; 2] = (&a[i..i + LANES]).try_into().unwrap();
            let o2: &mut [f64; 2] = (&mut out[i..i + LANES]).try_into().unwrap();
            abs_f64x2::apply(a2, o2);
            i += LANES;
        }
        while i < n {
            out[i] = a[i].abs();
            i += 1;
        }
    }

    fn name(&self) -> &'static str {
        "neon"
    }
}

// =========================================================================
// Global kernel selection
// =========================================================================

static GLOBAL_KERNEL: OnceLock<Box<dyn SimdKernel>> = OnceLock::new();

/// Returns the best available [`SimdKernel`] for this process.
///
/// The first call probes CPU feature flags and selects the optimal
/// implementation; all subsequent calls return the cached pointer with a
/// single Acquire load.
///
/// Priority: AVX2 (`x86_64`) > NEON (aarch64) > scalar.
pub fn global_kernel() -> &'static dyn SimdKernel {
    GLOBAL_KERNEL.get_or_init(select_best_kernel).as_ref()
}

fn select_best_kernel() -> Box<dyn SimdKernel> {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::detect::has_avx2() {
        return Box::new(Avx2Kernel);
    }

    #[cfg(target_arch = "aarch64")]
    if crate::simd::detect::has_neon() {
        return Box::new(NeonKernel);
    }

    Box::new(ScalarKernel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_kernel_add() {
        let a = [1.0_f64, 2.0, 3.0, 4.0];
        let b = [10.0_f64; 4];
        let mut out = [0.0_f64; 4];
        ScalarKernel.batch_add(&a, &b, &mut out);
        assert_eq!(out, [11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn scalar_kernel_fma() {
        let a = [2.0_f64, 3.0];
        let b = [4.0_f64, 5.0];
        let c = [1.0_f64, 1.0];
        let mut out = [0.0_f64; 2];
        ScalarKernel.batch_fma(&a, &b, &c, &mut out);
        assert_eq!(out, [9.0, 16.0]);
    }

    #[test]
    fn global_kernel_returns_valid_name() {
        let name = global_kernel().name();
        assert!(!name.is_empty());
    }

    #[test]
    fn global_kernel_add_matches_scalar() {
        let a: Vec<f64> = (0..16).map(f64::from).collect();
        let b = vec![1.0_f64; 16];
        let mut out_global = vec![0.0_f64; 16];
        let mut out_scalar = vec![0.0_f64; 16];
        global_kernel().batch_add(&a, &b, &mut out_global);
        ScalarKernel.batch_add(&a, &b, &mut out_scalar);
        for i in 0..16 {
            assert!(
                (out_global[i] - out_scalar[i]).abs() < 1e-12,
                "mismatch at {i}"
            );
        }
    }
}
