//! GPU compilation and execution module.
//!
//! Provides WebGPU (via `wgpu`) JIT compilation of symbolic expressions
//! directly to WGSL (WebGPU Shading Language) compute shaders, alongside
//! zero-copy execution of parallel batch jobs on the GPU.

#[cfg(feature = "gpu")]
pub mod compiler;
