//! NEON 2-lane f64 division: `out[0..2] = lhs[0..2] / rhs[0..2]`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn apply_neon(lhs: &[f64; 2], rhs: &[f64; 2], out: &mut [f64; 2]) {
    use std::arch::aarch64::*;
    let a = vld1q_f64(lhs.as_ptr());
    let b = vld1q_f64(rhs.as_ptr());
    let c = vdivq_f64(a, b);
    vst1q_f64(out.as_mut_ptr(), c);
}

/// Divides two 2-element f64 arrays element-wise, using NEON on aarch64 when available.
pub fn apply(lhs: &[f64; 2], rhs: &[f64; 2], out: &mut [f64; 2]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { apply_neon(lhs, rhs, out) };
        }
    }
    out[0] = lhs[0] / rhs[0];
    out[1] = lhs[1] / rhs[1];
}
