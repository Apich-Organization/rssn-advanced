//! NEON 2-lane f64 negation: `out[0..2] = -inp[0..2]`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn apply_neon(inp: &[f64; 2], out: &mut [f64; 2]) {
    use std::arch::aarch64::*;
    let a = vld1q_f64(inp.as_ptr());
    let c = vnegq_f64(a);
    vst1q_f64(out.as_mut_ptr(), c);
}

/// Negates two f64 values, using NEON on aarch64 when available.
pub fn apply(inp: &[f64; 2], out: &mut [f64; 2]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { apply_neon(inp, out) };
        }
    }
    out[0] = -inp[0];
    out[1] = -inp[1];
}
