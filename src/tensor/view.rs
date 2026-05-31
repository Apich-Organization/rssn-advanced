//! Zero-copy tensor views over flat `f64` buffers.
//!
//! A `TensorView` pairs a raw read-only slice with its `Shape` and `Strides`,
//! enabling element access via multi-dimensional indexing without allocation.
//! A `TensorViewMut` provides the same over a mutable buffer.
//!
//! ## Safety
//!
//! All public indexing methods perform bounds checking by default.
//! The `get_unchecked` family is `unsafe` and requires the caller to guarantee
//! that the flat index does not exceed the buffer length.

use super::shape::{Shape, Strides};
use smallvec::SmallVec;

/// An immutable zero-copy view of a multi-dimensional `f64` tensor.
///
/// This struct is `Send` and `Sync` because all its components (`&[f64]`, `Shape`, `Strides`, and `usize`) are `Send` and `Sync`.
pub struct TensorView<'a> {
    /// Flat backing buffer of `f64` elements (the entire underlying buffer).
    data: &'a [f64],
    /// Logical shape (extents per dimension).
    shape: Shape,
    /// Element-offset per index step along each dimension.
    strides: Strides,
    /// Offset from the start of `data` to the first element of this view.
    storage_offset: usize,
}

impl<'a> TensorView<'a> {
    /// Constructs a new `TensorView`.
    ///
    /// # Panics
    /// Panics if the buffer is smaller than the maximum addressable index implied by
    /// the strides and storage offset.
    #[must_use]
    pub fn new(data: &'a [f64], shape: Shape, storage_offset: usize) -> Self {
        let strides = shape.row_major_strides();
        let max_idx = if shape.numel() > 0 {
            strides.max_flat_index(&shape)
        } else {
            0
        };
        let required_len = storage_offset + max_idx + if shape.numel() > 0 { 1 } else { 0 };
        assert!(
            data.len() >= required_len,
            "Buffer too small for contiguous shape: need {} elements, got {}",
            required_len,
            data.len()
        );

        Self {
            data,
            shape,
            strides,
            storage_offset,
        }
    }

    /// Constructs a new `TensorView` with default `storage_offset` of 0.
    #[must_use]
    pub fn new_default(data: &'a [f64], shape: Shape) -> Self {
        Self::new(data, shape, 0)
    }

    /// Constructs a view with explicit (possibly non-contiguous) strides.
    ///
    /// # Panics
    /// Panics if `shape.rank() != strides.as_slice().len()`, or if the backing
    /// buffer is too small to safely access the largest flat index addressable by these strides.
    #[must_use]
    pub fn with_strides(
        data: &'a [f64],
        shape: Shape,
        strides: Strides,
        storage_offset: usize,
    ) -> Self {
        assert_eq!(
            shape.rank(),
            strides.as_slice().len(),
            "Rank mismatch between shape and strides"
        );

        let max_idx = if shape.numel() > 0 {
            strides.max_flat_index(&shape)
        } else {
            0
        };
        let required_len = storage_offset + max_idx + if shape.numel() > 0 { 1 } else { 0 };
        assert!(
            data.len() >= required_len,
            "Buffer too small for strided layout: need space for index {} (required size {}), got buffer of size {}",
            max_idx,
            required_len,
            data.len()
        );

        Self {
            data,
            shape,
            strides,
            storage_offset,
        }
    }

    /// Constructs a new `TensorView` with explicit strides and default `storage_offset` of 0.
    #[must_use]
    pub fn with_strides_default(data: &'a [f64], shape: Shape, strides: Strides) -> Self {
        Self::with_strides(data, shape, strides, 0)
    }

    /// Returns a reference to the `Shape`.
    #[must_use]
    pub const fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns a reference to the `Strides`.
    #[must_use]
    pub const fn strides(&self) -> &Strides {
        &self.strides
    }

    /// Returns the element at the multi-dimensional index.
    ///
    /// Returns `None` if any index component is out of bounds.
    #[must_use]
    pub fn get(&self, idx: &[usize]) -> Option<f64> {
        if idx.len() != self.shape.rank() {
            return None;
        }
        for (&i, &d) in idx.iter().zip(self.shape.dims()) {
            if i >= d {
                return None;
            }
        }
        let flat = self.strides.flat_index(idx);
        self.data.get(self.storage_offset + flat).copied()
    }

    /// Returns the element at the multi-dimensional index without bounds checking.
    ///
    /// # Safety
    /// The caller must guarantee that `idx` is a valid multi-dimensional index
    /// for this tensor view.
    #[must_use]
    pub unsafe fn get_unchecked(&self, idx: &[usize]) -> f64 {
        let flat = self.strides.flat_index(idx);
        // Explicit unsafe block for the call to get_unchecked
        unsafe { *self.data.get_unchecked(self.storage_offset + flat) }
    }

    /// Returns the rank (number of dimensions).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Returns the total number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Returns `true` if this view has a contiguous layout.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        if self.shape.numel() <= 1 {
            return true;
        }
        self.strides == self.shape.row_major_strides()
    }

    /// Returns the flat backing buffer (the entire underlying memory).
    /// Use `as_slice()` to get the view's active segment.
    #[must_use]
    pub const fn as_raw_slice(&self) -> &[f64] {
        self.data
    }

    /// Returns the active slice of this view.
    ///
    /// # Panics
    /// Panics if the view is not contiguous.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        assert!(
            self.is_contiguous(),
            "View is not contiguous, cannot convert to slice."
        );
        &self.data[self.storage_offset..self.storage_offset + self.numel()]
    }

    /// Iterates over all elements in row-major order.
    ///
    /// This is the fundamental building block for element-wise operations and
    /// vectorized kernel dispatch.
    pub fn iter_elements(&self) -> ElementIter<'_> {
        if self.is_contiguous() {
            ElementIter::Contiguous(self.as_slice().iter())
        } else {
            let rank = self.shape.rank();
            let idx = SmallVec::<[usize; 8]>::from_elem(0, rank);
            let done = self.shape.dims().contains(&0);
            ElementIter::Strided(StridedElementIter {
                data: self.data,
                strides: self.strides.clone(),
                dims: self.shape.clone(),
                idx,
                done,
                flat_offset: self.storage_offset,
                elements_yielded: 0,
            })
        }
    }
}

impl<'a, const N: usize> core::ops::Index<[usize; N]> for TensorView<'a> {
    type Output = f64;

    #[inline]
    fn index(&self, index: [usize; N]) -> &Self::Output {
        assert_eq!(N, self.shape.rank(), "Index rank mismatch");
        for (&i, &d) in index.iter().zip(self.shape.dims()) {
            assert!(i < d, "Index {} out of bounds for dimension size {}", i, d);
        }
        let flat = self.strides.flat_index(&index);
        &self.data[self.storage_offset + flat]
    }
}

impl<'a> core::ops::Index<&[usize]> for TensorView<'a> {
    type Output = f64;

    #[inline]
    fn index(&self, index: &[usize]) -> &Self::Output {
        assert_eq!(index.len(), self.shape.rank(), "Index rank mismatch");
        for (&i, &d) in index.iter().zip(self.shape.dims()) {
            assert!(i < d, "Index {} out of bounds for dimension size {}", i, d);
        }
        let flat = self.strides.flat_index(index);
        &self.data[self.storage_offset + flat]
    }
}

impl<'a> core::ops::Index<(usize,)> for TensorView<'a> {
    type Output = f64;
    #[inline]
    fn index(&self, index: (usize,)) -> &Self::Output {
        &self[[index.0]]
    }
}

impl<'a> core::ops::Index<(usize, usize)> for TensorView<'a> {
    type Output = f64;
    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self[[index.0, index.1]]
    }
}

impl<'a> core::ops::Index<(usize, usize, usize)> for TensorView<'a> {
    type Output = f64;
    #[inline]
    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        &self[[index.0, index.1, index.2]]
    }
}

/// A mutable zero-copy view of a multi-dimensional `f64` tensor.
///
/// This struct is `Send` but NOT `Sync`. It is `Send` because `&mut [f64]` is `Send`.
/// It is not `Sync` because `&mut [f64]` is not `Sync`.
pub struct TensorViewMut<'a> {
    data: &'a mut [f64],
    shape: Shape,
    strides: Strides,
    storage_offset: usize,
}

impl<'a> TensorViewMut<'a> {
    /// Constructs a new mutable `TensorViewMut`.
    ///
    /// # Panics
    /// Panics if the buffer is smaller than the maximum addressable flat index.
    #[must_use]
    pub fn new(data: &'a mut [f64], shape: Shape, storage_offset: usize) -> Self {
        let strides = shape.row_major_strides();
        let max_idx = if shape.numel() > 0 {
            strides.max_flat_index(&shape)
        } else {
            0
        };
        let required_len = storage_offset + max_idx + if shape.numel() > 0 { 1 } else { 0 };
        assert!(
            data.len() >= required_len,
            "Buffer too small for shape and offset: need {} elements, got {}",
            required_len,
            data.len(),
        );

        Self {
            data,
            shape,
            strides,
            storage_offset,
        }
    }

    /// Constructs a new `TensorViewMut` with default `storage_offset` of 0.
    #[must_use]
    pub fn new_default(data: &'a mut [f64], shape: Shape) -> Self {
        Self::new(data, shape, 0)
    }

    /// Returns a reference to the `Shape`.
    #[must_use]
    pub const fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns a reference to the `Strides`.
    #[must_use]
    pub const fn strides(&self) -> &Strides {
        &self.strides
    }

    /// Returns `true` if this view has a contiguous layout.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        if self.shape.numel() <= 1 {
            return true;
        }
        self.strides == self.shape.row_major_strides()
    }

    /// Returns the element at the multi-dimensional index mutably.
    ///
    /// Returns `None` if any index component is out of bounds.
    pub fn get_mut(&mut self, idx: &[usize]) -> Option<&mut f64> {
        if idx.len() != self.shape.rank() {
            return None;
        }
        for (&i, &d) in idx.iter().zip(self.shape.dims()) {
            if i >= d {
                return None;
            }
        }
        let flat = self.strides.flat_index(idx);
        self.data.get_mut(self.storage_offset + flat)
    }

    /// Returns a mutable reference to the element at the multi-dimensional index
    /// without bounds checking.
    ///
    /// # Safety
    /// The caller must guarantee that `idx` is a valid multi-dimensional index
    /// for this tensor view.
    #[must_use]
    pub unsafe fn get_unchecked_mut(&mut self, idx: &[usize]) -> &mut f64 {
        let flat = self.strides.flat_index(idx);
        unsafe { self.data.get_unchecked_mut(self.storage_offset + flat) }
    }

    /// Sets the element at the multi-dimensional index.
    ///
    /// Returns `false` if the index is out-of-bounds.
    pub fn set(&mut self, idx: &[usize], val: f64) -> bool {
        if let Some(slot) = self.get_mut(idx) {
            *slot = val;
            true
        } else {
            false
        }
    }

    /// Returns the flat backing buffer immutably (the entire underlying memory).
    /// Use `as_slice()` to get the view's active segment.
    #[must_use]
    pub const fn as_raw_slice(&self) -> &[f64] {
        self.data
    }

    /// Returns the flat backing buffer mutably (the entire underlying memory).
    /// Use `as_slice_mut()` to get the view's active segment.
    pub fn as_raw_slice_mut(&mut self) -> &mut [f64] {
        self.data
    }

    /// Returns the active slice of this view.
    ///
    /// # Panics
    /// Panics if the view is not contiguous.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        assert!(
            self.is_contiguous(),
            "View is not contiguous, cannot convert to slice."
        );
        &self.data[self.storage_offset..self.storage_offset + self.shape.numel()]
    }

    /// Returns the active mutable slice of this view.
    ///
    /// # Panics
    /// Panics if the view is not contiguous.
    pub fn as_slice_mut(&mut self) -> &mut [f64] {
        assert!(
            self.is_contiguous(),
            "View is not contiguous, cannot convert to mutable slice."
        );
        let numel = self.shape.numel();
        &mut self.data[self.storage_offset..self.storage_offset + numel]
    }

    /// Returns a read-only view of this mutable tensor.
    #[must_use]
    pub fn as_view(&self) -> TensorView<'_> {
        TensorView {
            data: self.data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            storage_offset: self.storage_offset,
        }
    }
}

impl<'a, const N: usize> core::ops::Index<[usize; N]> for TensorViewMut<'a> {
    type Output = f64;

    #[inline]
    fn index(&self, index: [usize; N]) -> &Self::Output {
        assert_eq!(N, self.shape.rank(), "Index rank mismatch");
        for (&i, &d) in index.iter().zip(self.shape.dims()) {
            assert!(i < d, "Index {} out of bounds for dimension size {}", i, d);
        }
        let flat = self.strides.flat_index(&index);
        &self.data[self.storage_offset + flat]
    }
}

impl<'a, const N: usize> core::ops::IndexMut<[usize; N]> for TensorViewMut<'a> {
    #[inline]
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        assert_eq!(N, self.shape.rank(), "Index rank mismatch");
        for (&i, &d) in index.iter().zip(self.shape.dims()) {
            assert!(i < d, "Index {} out of bounds for dimension size {}", i, d);
        }
        let flat = self.strides.flat_index(&index);
        &mut self.data[self.storage_offset + flat]
    }
}

impl<'a> core::ops::IndexMut<&[usize]> for TensorViewMut<'a> {
    #[inline]
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        assert_eq!(index.len(), self.shape.rank(), "Index rank mismatch");
        for (&i, &d) in index.iter().zip(self.shape.dims()) {
            assert!(i < d, "Index {} out of bounds for dimension size {}", i, d);
        }
        let flat = self.strides.flat_index(index);
        &mut self.data[self.storage_offset + flat]
    }
}

impl<'a> core::ops::Index<&[usize]> for TensorViewMut<'a> {
    type Output = f64;

    #[inline]
    fn index(&self, index: &[usize]) -> &Self::Output {
        assert_eq!(index.len(), self.shape.rank(), "Index rank mismatch");
        for (&i, &d) in index.iter().zip(self.shape.dims()) {
            assert!(i < d, "Index {} out of bounds for dimension size {}", i, d);
        }
        let flat = self.strides.flat_index(index);
        &self.data[self.storage_offset + flat]
    }
}

impl<'a> core::ops::Index<(usize,)> for TensorViewMut<'a> {
    type Output = f64;
    #[inline]
    fn index(&self, index: (usize,)) -> &Self::Output {
        &self[[index.0]]
    }
}

impl<'a> core::ops::Index<(usize, usize)> for TensorViewMut<'a> {
    type Output = f64;
    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self[[index.0, index.1]]
    }
}

impl<'a> core::ops::Index<(usize, usize, usize)> for TensorViewMut<'a> {
    type Output = f64;
    #[inline]
    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        &self[[index.0, index.1, index.2]]
    }
}

impl<'a> core::ops::IndexMut<(usize,)> for TensorViewMut<'a> {
    #[inline]
    fn index_mut(&mut self, index: (usize,)) -> &mut Self::Output {
        &mut self[[index.0]]
    }
}

impl<'a> core::ops::IndexMut<(usize, usize)> for TensorViewMut<'a> {
    #[inline]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self[[index.0, index.1]]
    }
}

impl<'a> core::ops::IndexMut<(usize, usize, usize)> for TensorViewMut<'a> {
    #[inline]
    fn index_mut(&mut self, index: (usize, usize, usize)) -> &mut Self::Output {
        &mut self[[index.0, index.1, index.2]]
    }
}

/// Element-wise binary operation on two broadcast-compatible views into an output.
///
/// This is the core primitive for all binary tensor operators (+, -, *, /).
/// Broadcasting is handled via the stride-zero convention: a dimension with
/// size 1 has stride 0, so the same element is reused without copying.
///
/// # Safety
/// To prevent undefined behavior, the caller must ensure that the output buffer window
/// does not overlap with the backing storage buffers of `a` and `b` unless an exact
/// in-place operation is intended and safe.
///
/// # Errors
/// Returns `Err(String)` if the shapes are not broadcast-compatible or the
/// output buffer is too small for the broadcast output shape.
pub fn broadcast_elementwise<F>(
    a: &TensorView<'_>,
    b: &TensorView<'_>,
    out: &mut TensorViewMut<'_>,
    op: F,
) -> Result<(), String>
where
    F: Fn(f64, f64) -> f64 + Copy,
{
    let out_shape = a.shape().broadcast_output(b.shape()).ok_or_else(|| {
        format!(
            "Shapes {} and {} are not broadcast-compatible",
            a.shape(),
            b.shape(),
        )
    })?;

    if out.shape() != &out_shape {
        return Err(format!(
            "Output shape {} doesn't match expected broadcast shape {}",
            out.shape(),
            out_shape,
        ));
    }

    let is_contiguous_fast_path = a.shape() == b.shape()
        && a.shape() == out.shape()
        && a.is_contiguous()
        && b.is_contiguous()
        && out.is_contiguous();

    let a_slice = a.as_raw_slice();
    let b_slice = b.as_raw_slice();

    if is_contiguous_fast_path {
        let out_slice = out.as_slice_mut();
        let len = a.numel();

        let a_sub = &a_slice[a.storage_offset..a.storage_offset + len];
        let b_sub = &b_slice[b.storage_offset..b.storage_offset + len];
        let out_sub = &mut out_slice[..len];

        for i in 0..len {
            out_sub[i] = op(a_sub[i], b_sub[i]);
        }
        return Ok(());
    }

    let rank = out_shape.rank();
    let a_offset = rank.saturating_sub(a.rank());
    let b_offset = rank.saturating_sub(b.rank());

    let a_strides = a.strides().as_slice();
    let b_strides = b.strides().as_slice();

    let mut a_strides_padded = SmallVec::<[usize; 8]>::from_elem(0, rank);
    let mut b_strides_padded = SmallVec::<[usize; 8]>::from_elem(0, rank);

    // If an original dimension size is 1, its broadcast stride is 0
    let a_dims = a.shape().dims();
    for i in a_offset..rank {
        if a_dims[i - a_offset] > 1 {
            a_strides_padded[i] = a_strides[i - a_offset];
        }
    }

    let b_dims = b.shape().dims();
    for i in b_offset..rank {
        if b_dims[i - b_offset] > 1 {
            b_strides_padded[i] = b_strides[i - b_offset];
        }
    }

    let out_strides_cloned = out.strides().clone();
    let output_iter = OutputFlatIndexIterator::new(
        out_shape.clone(),
        out_strides_cloned.clone(),
        a_strides_padded.clone(),
        b_strides_padded.clone(),
    );

    let a_step = if rank > 0 {
        a_strides_padded[rank - 1]
    } else {
        0
    };
    let b_step = if rank > 0 {
        b_strides_padded[rank - 1]
    } else {
        0
    };
    let out_step = if rank > 0 {
        out_strides_cloned.as_slice()[rank - 1]
    } else {
        0
    };

    let a_ptr = unsafe { a_slice.as_ptr().add(a.storage_offset) };
    let b_ptr = unsafe { b_slice.as_ptr().add(b.storage_offset) };
    let out_ptr = unsafe { out.as_raw_slice_mut().as_mut_ptr().add(out.storage_offset) };

    // Debug assertions to catch physical overlap/aliasing in development
    debug_assert!(
        a_ptr as *const () != out_ptr as *const (),
        "Tensor A and Output alias!"
    );
    debug_assert!(
        b_ptr as *const () != out_ptr as *const (),
        "Tensor B and Output alias!"
    );

    unsafe {
        broadcast_elementwise_kernel(
            output_iter,
            a_ptr,
            b_ptr,
            out_ptr,
            a_step,
            b_step,
            out_step,
            op,
        );
    }

    Ok(())
}

#[inline(always)]
unsafe fn process_row_contiguous<F>(
    a_ptr: *const f64,
    b_ptr: *const f64,
    out_ptr: *mut f64,
    mut current_a: usize,
    mut current_b: usize,
    mut current_out: usize,
    row_len: usize,
    op: F,
) where
    F: Fn(f64, f64) -> f64,
{
    for _ in 0..row_len {
        let va = unsafe { *a_ptr.add(current_a) };
        let vb = unsafe { *b_ptr.add(current_b) };
        unsafe { *out_ptr.add(current_out) = op(va, vb) };
        current_a += 1;
        current_b += 1;
        current_out += 1;
    }
}

#[inline(always)]
#[allow(clippy::similar_names)]
unsafe fn process_row_broadcast_a<OpF>(
    a_ptr: *const f64,
    b_ptr: *const f64,
    out_ptr: *mut f64,
    current_a_fixed: usize,
    mut current_b: usize,
    mut current_out: usize,
    b_step: usize,
    out_step: usize,
    row_len: usize,
    op: OpF,
) where
    OpF: Fn(f64, f64) -> f64,
{
    let va = unsafe { *a_ptr.add(current_a_fixed) };
    for _ in 0..row_len {
        let vb = unsafe { *b_ptr.add(current_b) };
        unsafe { *out_ptr.add(current_out) = op(va, vb) };
        current_b += b_step;
        current_out += out_step;
    }
}

#[inline(always)]
#[allow(clippy::similar_names)]
unsafe fn process_row_broadcast_b<OpF>(
    a_ptr: *const f64,
    b_ptr: *const f64,
    out_ptr: *mut f64,
    mut current_a: usize,
    current_b_fixed: usize,
    mut current_out: usize,
    a_step: usize,
    out_step: usize,
    row_len: usize,
    op: OpF,
) where
    OpF: Fn(f64, f64) -> f64,
{
    let vb = unsafe { *b_ptr.add(current_b_fixed) };
    for _ in 0..row_len {
        let va = unsafe { *a_ptr.add(current_a) };
        unsafe { *out_ptr.add(current_out) = op(va, vb) };
        current_a += a_step;
        current_out += out_step;
    }
}

#[inline(always)]
#[allow(clippy::similar_names)]
unsafe fn process_row_broadcast_ab<OpF>(
    a_ptr: *const f64,
    b_ptr: *const f64,
    out_ptr: *mut f64,
    current_a_fixed: usize,
    current_b_fixed: usize,
    mut current_out: usize,
    out_step: usize,
    row_len: usize,
    op: OpF,
) where
    OpF: Fn(f64, f64) -> f64,
{
    let va = unsafe { *a_ptr.add(current_a_fixed) };
    let vb = unsafe { *b_ptr.add(current_b_fixed) };
    let result = op(va, vb);
    for _ in 0..row_len {
        unsafe { *out_ptr.add(current_out) = result };
        current_out += out_step;
    }
}

#[inline(always)]
unsafe fn process_row_general<F>(
    a_ptr: *const f64,
    b_ptr: *const f64,
    out_ptr: *mut f64,
    mut current_a: usize,
    mut current_b: usize,
    mut current_out: usize,
    a_step: usize,
    b_step: usize,
    out_step: usize,
    row_len: usize,
    op: F,
) where
    F: Fn(f64, f64) -> f64,
{
    for _ in 0..row_len {
        let va = unsafe { *a_ptr.add(current_a) };
        let vb = unsafe { *b_ptr.add(current_b) };
        unsafe { *out_ptr.add(current_out) = op(va, vb) };
        current_a += a_step;
        current_b += b_step;
        current_out += out_step;
    }
}

#[inline(always)]
unsafe fn broadcast_elementwise_kernel<F>(
    output_iter: OutputFlatIndexIterator,
    a_ptr: *const f64,
    b_ptr: *const f64,
    out_ptr: *mut f64,
    a_step: usize,
    b_step: usize,
    out_step: usize,
    op: F,
) where
    F: Fn(f64, f64) -> f64 + Copy,
{
    for (a_flat, b_flat, out_flat, row_len) in output_iter {
        if a_step == 1 && b_step == 1 && out_step == 1 {
            unsafe {
                process_row_contiguous(
                    a_ptr, b_ptr, out_ptr, a_flat, b_flat, out_flat, row_len, op,
                );
            }
        } else if a_step == 0 && b_step == 0 {
            unsafe {
                process_row_broadcast_ab(
                    a_ptr, b_ptr, out_ptr, a_flat, b_flat, out_flat, out_step, row_len, op,
                );
            }
        } else if a_step == 0 {
            unsafe {
                process_row_broadcast_a(
                    a_ptr, b_ptr, out_ptr, a_flat, b_flat, out_flat, b_step, out_step, row_len, op,
                );
            }
        } else if b_step == 0 {
            unsafe {
                process_row_broadcast_b(
                    a_ptr, b_ptr, out_ptr, a_flat, b_flat, out_flat, a_step, out_step, row_len, op,
                );
            }
        } else {
            unsafe {
                process_row_general(
                    a_ptr, b_ptr, out_ptr, a_flat, b_flat, out_flat, a_step, b_step, out_step,
                    row_len, op,
                );
            }
        }
    }
}

pub enum ElementIter<'a> {
    Contiguous(std::slice::Iter<'a, f64>),
    Strided(StridedElementIter<'a>),
}

impl<'a> Iterator for ElementIter<'a> {
    type Item = f64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Contiguous(it) => it.next().copied(),
            Self::Strided(it) => it.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Contiguous(it) => it.size_hint(),
            Self::Strided(it) => it.size_hint(),
        }
    }
}

/// An optimized element iterator over strided tensors.
///
/// Iteration tracking coordinates and memory offsets are modified incrementally
/// using an amortized $O(1)$ technique, resolving performance penalties when
/// performing element-by-element iteration.
pub struct StridedElementIter<'a> {
    data: &'a [f64],
    strides: Strides,
    dims: Shape,
    idx: SmallVec<[usize; 8]>,
    done: bool,
    flat_offset: usize,
    elements_yielded: usize,
}

impl<'a> Iterator for StridedElementIter<'a> {
    type Item = f64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let val = unsafe { *self.data.get_unchecked(self.flat_offset) };
        self.elements_yielded += 1;

        let rank = self.dims.rank();
        let dims = self.dims.dims();
        let strides_slice = self.strides.as_slice();

        let mut carry = true;
        for i in (0..rank).rev() {
            self.idx[i] += 1;
            self.flat_offset += strides_slice[i];

            if self.idx[i] >= dims[i] {
                // Rewind flat memory offset accumulated along this dimension axis
                let steps_taken = self.idx[i];
                self.flat_offset -= strides_slice[i] * steps_taken;
                self.idx[i] = 0;
            } else {
                carry = false;
                break;
            }
        }

        if carry {
            self.done = true;
        }

        Some(val)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done {
            (0, Some(0))
        } else {
            let total_numel = self.dims.numel();
            let remaining = total_numel.saturating_sub(self.elements_yielded);
            (remaining, Some(remaining))
        }
    }
}

/// High-performance flat offset iterator resolving broadcast targets coordinate-by-coordinate.
pub struct OutputFlatIndexIterator {
    dims: Shape,
    strides: Strides,
    idx: SmallVec<[usize; 8]>,
    done: bool,
    a_strides_padded: SmallVec<[usize; 8]>,
    b_strides_padded: SmallVec<[usize; 8]>,
    current_a: usize,
    current_b: usize,
    current_out: usize,
}

impl OutputFlatIndexIterator {
    #[must_use]
    pub fn new(
        output_shape: Shape,
        output_strides: Strides,
        a_strides_padded: SmallVec<[usize; 8]>,
        b_strides_padded: SmallVec<[usize; 8]>,
    ) -> Self {
        let rank = output_shape.rank();
        let idx = SmallVec::<[usize; 8]>::from_elem(0, rank);
        let done = output_shape.numel() == 0;

        let mut current_a = 0;
        let mut current_b = 0;
        let mut current_out = 0;

        if rank > 0 && !done {
            let out_strides_slice = output_strides.as_slice();
            for i in 0..rank {
                current_a += idx[i] * a_strides_padded[i];
                current_b += idx[i] * b_strides_padded[i];
                current_out += idx[i] * out_strides_slice[i];
            }
        }

        OutputFlatIndexIterator {
            dims: output_shape,
            strides: output_strides,
            idx,
            done,
            a_strides_padded,
            b_strides_padded,
            current_a,
            current_b,
            current_out,
        }
    }
}

impl Iterator for OutputFlatIndexIterator {
    type Item = (usize, usize, usize, usize);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let rank = self.dims.rank();
        let dims = self.dims.dims();
        let out_strides_slice = self.strides.as_slice();

        if rank == 0 {
            self.done = true;
            return Some((0, 0, 0, 1));
        }

        let innermost_dim_idx = rank - 1;
        let row_len = dims[innermost_dim_idx];

        let res = (self.current_a, self.current_b, self.current_out, row_len);

        let mut carry = true;
        let mut i = rank;

        while carry && i > 0 {
            i -= 1;

            if i == innermost_dim_idx {
                self.idx[i] += row_len;
                self.current_a += self.a_strides_padded[i] * row_len;
                self.current_b += self.b_strides_padded[i] * row_len;
                self.current_out += out_strides_slice[i] * row_len;
            } else {
                self.idx[i] += 1;
                self.current_a += self.a_strides_padded[i];
                self.current_b += self.b_strides_padded[i];
                self.current_out += out_strides_slice[i];
            }

            if self.idx[i] >= dims[i] {
                let steps_taken = self.idx[i];
                // Subtract the accumulated steps to rewind the offsets
                self.current_a -= self.a_strides_padded[i] * steps_taken;
                self.current_b -= self.b_strides_padded[i] * steps_taken;
                self.current_out -= out_strides_slice[i] * steps_taken;
                self.idx[i] = 0;
            } else {
                carry = false;
            }
        }

        if carry {
            self.done = true;
        }

        Some(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_view_indexing() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let view = TensorView::new_default(&data, Shape::matrix(2, 3));
        assert_eq!(view.get(&[0, 0]), Some(0.0));
        assert_eq!(view.get(&[0, 2]), Some(2.0));
        assert_eq!(view.get(&[1, 1]), Some(4.0));
        assert_eq!(view.get(&[2, 0]), None); // out of bounds

        assert_eq!(view[[0, 0]], 0.0);
        assert_eq!(view[[0, 2]], 2.0);
        assert_eq!(view[[1, 1]], 4.0);
    }

    #[test]
    fn test_tensor_view_mut_indexing() {
        let mut data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut view_mut = TensorViewMut::new_default(&mut data, Shape::matrix(2, 3));

        assert_eq!(view_mut[[1, 1]], 4.0);
        view_mut[[1, 1]] = 42.0;
        assert_eq!(view_mut[[1, 1]], 42.0);

        assert!(view_mut.set(&[0, 1], 99.0));
        assert_eq!(view_mut[[0, 1]], 99.0);
    }
}
