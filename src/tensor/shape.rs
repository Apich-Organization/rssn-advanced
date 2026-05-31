//! Shape and stride metadata for multi-dimensional tensors.
//!
//! A `Shape` is an ordered list of dimension extents (e.g. `[3, 4]` for a
//! 3×4 matrix). A `Strides` is the corresponding list of byte-offsets (in
//! *elements*, not bytes) that map a multi-index to a flat memory offset.
//!
//! ## Broadcasting
//!
//! Following `NumPy` semantics, a dimension of size 1 can be broadcast against
//! any larger dimension. The stride for that dimension is set to `0`, so the
//! same element is reused for every index along that axis without any copying.
//!
//! ## Row-major (C-order) layout
//!
//! By default `Shape::row_major_strides()` computes C-order strides so that
//! the last axis is contiguous. Column-major (Fortran/BLAS) strides can be
//! constructed manually for interop with BLAS calls.

use smallvec::SmallVec;

/// The rank (number of dimensions) limit stored inline before heap-spilling.
const INLINE_RANK: usize = 8;

/// Dimension extents for a tensor (e.g. `[batch, rows, cols]`).
///
/// Stored inline (stack-allocated) for rank ≤ 8, heap-allocated otherwise.
///
/// This struct is `Send` and `Sync` because all its components (`usize` and `smallvec::SmallVec`) are `Send` and `Sync`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    /// Inline storage for up to `INLINE_RANK` dimensions.
    dims: smallvec::SmallVec<[usize; INLINE_RANK]>,
}

impl Shape {
    /// Constructs a `Shape` from a slice of dimension extents.
    #[must_use]
    pub fn from_dims(dims: &[usize]) -> Self {
        Self {
            dims: dims.iter().copied().collect(),
        }
    }

    /// Returns the number of dimensions (rank).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Returns the total number of elements (product of all dimensions).
    ///
    /// Returns `1` for a scalar (rank-0) shape.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Returns a slice of dimension extents.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Computes row-major (C-order) strides for this shape.
    ///
    /// For a shape `[d0, d1, d2, …, dn]` the strides are
    /// `[d1*d2*…*dn, d2*…*dn, …, dn, 1]`.
    ///
    /// Broadcasting dimensions (size == 1) receive stride `0` so that their
    /// element is reused without copying.
    #[must_use]
    #[allow(clippy::unnecessary_cast)]
    pub fn row_major_strides(&self) -> Strides {
        let rank = self.dims.len();
        if rank == 0 {
            return Strides {
                strides: smallvec::SmallVec::new(),
            };
        }

        // First, compute standard row-major strides without considering broadcasting
        let mut actual_strides = smallvec::smallvec![0usize; rank];
        actual_strides[rank - 1] = 1; // Last dimension has stride 1
        for i in (0..rank - 1).rev() {
            // Stride for current dimension is (stride of next dimension) * (size of next dimension)
            let (result, overflow) =
                (actual_strides[i + 1] as usize).overflowing_mul(self.dims[i + 1]);
            actual_strides[i] = if overflow { usize::MAX } else { result };
        }

        // Then, apply broadcast zeroing for dimensions of size 1
        let mut final_strides = actual_strides;
        for (i, &d) in self.dims.iter().enumerate() {
            if d == 1 {
                final_strides[i] = 0;
            }
        }
        Strides {
            strides: final_strides,
        }
    }

    /// Returns `true` if the two shapes are broadcast-compatible.
    ///
    /// Shapes are broadcast-compatible if, after right-aligning them, every
    /// pair of dimensions is either equal or one of them is 1.
    #[must_use]
    pub fn broadcast_compatible(&self, other: &Self) -> bool {
        let a = self.dims();
        let b = other.dims();
        let len = a.len().max(b.len());
        for i in 0..len {
            let dim_idx_from_right = len - 1 - i;
            let da = if dim_idx_from_right < a.len() {
                a[a.len() - 1 - dim_idx_from_right]
            } else {
                1
            };
            let db = if dim_idx_from_right < b.len() {
                b[b.len() - 1 - dim_idx_from_right]
            } else {
                1
            };
            if da != db && da != 1 && db != 1 {
                return false;
            }
        }
        true
    }

    /// Returns the output shape after broadcasting two shapes together.
    ///
    /// # Errors
    /// Returns `None` if the shapes are not broadcast-compatible.
    #[must_use]
    pub fn broadcast_output(&self, other: &Self) -> Option<Self> {
        let a = self.dims();
        let b = other.dims();
        let len = a.len().max(b.len());
        let mut out = smallvec::SmallVec::<[usize; INLINE_RANK]>::with_capacity(len);

        for i in 0..len {
            // Calculate the corresponding index for a and b by left-padding with 1s
            let da = if i + a.len() < len {
                1
            } else {
                a[i + a.len() - len]
            };
            let db = if i + b.len() < len {
                1
            } else {
                b[i + b.len() - len]
            };

            if da != db && da != 1 && db != 1 {
                return None; // Merged compatibility check here
            }
            out.push(da.max(db));
        }
        Some(Self { dims: out })
    }

    /// Returns a scalar (rank-0) shape.
    #[must_use]
    pub fn scalar() -> Self {
        Self {
            dims: smallvec::SmallVec::new(),
        }
    }

    /// Returns a 1-D shape with `n` elements.
    #[must_use]
    pub fn vector(n: usize) -> Self {
        Self::from_dims(&[n])
    }

    /// Returns a 2-D matrix shape.
    #[must_use]
    pub fn matrix(rows: usize, cols: usize) -> Self {
        Self::from_dims(&[rows, cols])
    }
}

impl core::fmt::Display for Shape {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

/// Stride values (element-offset per index step) for each dimension.
///
/// A stride of `0` means broadcast: the same element is reused across that axis.
///
/// This struct is `Send` and `Sync` because all its components (`usize` and `smallvec::SmallVec`) are `Send` and `Sync`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Strides {
    strides: smallvec::SmallVec<[usize; INLINE_RANK]>,
}

impl Strides {
    /// Returns a slice of stride values.
    #[must_use]
    pub fn as_slice(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the stride for the given dimension index.
    #[must_use]
    pub fn get(&self, dim: usize) -> Option<usize> {
        self.strides.get(dim).copied()
    }

    /// Computes a flat element index from a multi-dimensional index.
    /// Returns `Some(flat_index)` if the calculation does not overflow `usize`,
    /// `None` otherwise.
    #[must_use]
    pub fn checked_flat_index(&self, idx: &[usize]) -> Option<usize> {
        debug_assert_eq!(
            idx.len(),
            self.strides.len(),
            "Rank mismatch for checked_flat_index"
        );
        let mut flat = 0usize;
        for (&i, &s) in idx.iter().zip(self.strides.iter()) {
            let term = i.checked_mul(s)?; // Check for overflow on multiplication
            flat = flat.checked_add(term)?; // Check for overflow on summation
        }
        Some(flat)
    }

    /// Computes a flat element index from a multi-dimensional index.
    ///
    /// # Panics
    /// Panics if `idx.len() != strides.len()` (rank mismatch) or if the flat index calculation overflows `usize`.
    #[must_use]
    pub fn flat_index(&self, idx: &[usize]) -> usize {
        assert_eq!(idx.len(), self.strides.len(), "Index rank mismatch");
        self.checked_flat_index(idx)
            .expect("Flat index calculation overflowed. This indicates an invalid tensor configuration or an out-of-bounds access that would result in an extremely large flat index.")
    }

    /// Calculates the maximum possible flat index for these strides given a shape.
    /// Returns the maximum flat index. Panics if the calculation overflows `usize`.
    ///
    /// # Panics
    /// Panics if the rank of `self` and `shape` do not match.
    /// Panics if the maximum flat index calculation overflows `usize`.
    #[must_use]
    pub fn max_flat_index(&self, shape: &Shape) -> usize {
        assert_eq!(
            self.strides.len(),
            shape.rank(),
            "Rank mismatch for Strides::max_flat_index"
        );
        let max_idx_components: SmallVec<[usize; INLINE_RANK]> =
            shape.dims().iter().map(|&d| d.saturating_sub(1)).collect();
        self.checked_flat_index(&max_idx_components)
            .expect("Maximum flat index calculation overflowed. This indicates an invalid strides/shape configuration that could lead to out-of-bounds access or silent data corruption.")
    }
} // Closes impl Strides
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_major_strides_2d() {
        let s = Shape::matrix(3, 4);
        let strides = s.row_major_strides();
        assert_eq!(strides.as_slice(), &[4, 1]);
    }

    #[test]
    fn test_broadcast_strides() {
        let s = Shape::from_dims(&[1, 4]);
        let strides = s.row_major_strides();
        // Dimension 0 is broadcast (size=1), so stride[0] == 0
        assert_eq!(strides.as_slice()[0], 0);
        assert_eq!(strides.as_slice()[1], 1);
    }

    #[test]
    fn test_broadcast_output() {
        let a = Shape::from_dims(&[3, 1]);
        let b = Shape::from_dims(&[1, 4]);
        let out = a.broadcast_output(&b).unwrap();
        assert_eq!(out.dims(), &[3, 4]);
    }

    #[test]
    fn test_numel() {
        let s = Shape::matrix(3, 4);
        assert_eq!(s.numel(), 12);
    }

    #[test]
    fn test_flat_index() {
        let s = Shape::matrix(3, 4);
        let strides = s.row_major_strides();
        // Element [1, 2] in a 3×4 matrix is at flat offset 1*4 + 2*1 = 6
        assert_eq!(strides.flat_index(&[1, 2]), 6);
    }
}
