//! Multi-dimensional tensor support: Shape, Stride, and TensorView.
//!
//! This module provides the foundational data types for multi-dimensional
//! tensor operations. It is intentionally separate from the JIT compiler so
//! that shape/stride metadata can be used independently of Cranelift.

pub mod shape;
pub mod view;
