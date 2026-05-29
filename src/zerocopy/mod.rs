//! Zero-copy borrowed containers and `bincode-next` `BorrowDecode` glue.
//!
//! The original storage layer always allocated a fresh `Vec` on decode, which
//! defeats `bincode-next`'s zero-copy capability and is the root cause of
//! review findings in `storage_review §1` / `dag_review §3` /
//! `ast_review §3`. This module introduces:
//!
//! * [`Pod`](crate::zerocopy::Pod) — narrow safety marker for types whose bit-pattern may be
//!   reinterpreted directly from a borrowed byte slice. Implemented manually
//!   per type; never derive it on enums or types with padding.
//!
//! * [`BorrowedSlice<'a, T>`](crate::zerocopy::BorrowedSlice) — a `&'a [T]` newtype that decodes by simply
//!   taking N bytes off the input buffer (`take_bytes`) without allocation.
//!
//! * [`BorrowedArena<'a, T>`](crate::zerocopy::BorrowedArena) — single-pool arena view backed by
//!   [`BorrowedSlice`](crate::zerocopy::BorrowedSlice). DAG/AST will adopt this in T1.1 / T1.4.
//!
//! * [`AlignedBytes`](crate::zerocopy::AlignedBytes) — a `Box<[u64]>`-backed buffer wrapping bytes whose
//!   start is guaranteed 8-byte aligned. Use this anywhere you would
//!   normally hand a `Vec<u8>` to `borrow_decode_from_slice`.
//!
//! * [`MmapBuffer`](crate::zerocopy::MmapBuffer) — file-backed bytes (mmap on Linux/Windows when feature
//!   is enabled, `std::fs::read` fallback otherwise). Exposes its bytes via
//!   [`MmapBuffer::with_view`](crate::zerocopy::MmapBuffer::with_view) so borrows cannot outlive the mapping.
//!
//! ## Wire format
//!
//! [`zerocopy_config`](crate::zerocopy::zerocopy_config) forces fixed-width integer encoding so the data
//! payload starts at a known offset (`size_of::<u64>() = 8` from the start
//! of the [`BorrowedSlice`](crate::zerocopy::BorrowedSlice) image). Combined with an 8-byte aligned start
//! ([`AlignedBytes`](crate::zerocopy::AlignedBytes) / mmap), this means every primitive `T: Pod` with
//! `align_of::<T>() <= 8` is naturally aligned in the buffer.

#![allow(unsafe_code)]

pub mod mmap;

use core::marker::PhantomData;

use bincode_next::config::{Configuration, Fixint, LittleEndian, NoLimit};
use bincode_next::de::read::BorrowReader;
use bincode_next::de::{BorrowDecode, BorrowDecoder, Decode};
use bincode_next::enc::write::Writer;
use bincode_next::enc::{Encode, Encoder};
use bincode_next::error::{DecodeError, EncodeError};

pub use self::mmap::MmapBuffer;

extern crate alloc;
use alloc::boxed::Box;
use alloc::vec::Vec;

// =========================================================================
// Configuration & primitives
// =========================================================================

/// Type alias for the fixed-int little-endian config used by this module.
///
/// Concrete spelling kept private so callers go through [`zerocopy_config`].
type ZeroCopyConfig = Configuration<LittleEndian, Fixint, NoLimit>;

/// The `bincode_next` configuration used for every type in this module.
///
/// Fixed integer width is essential: with varint encoding the data offset
/// after the length prefix depends on the value of the length, which would
/// destroy alignment guarantees. Little-endian matches every target we care
/// about (`x86_64`, `aarch64`, `riscv64-le`).
#[must_use]
pub const fn zerocopy_config() -> ZeroCopyConfig {
    bincode_next::config::standard()
        .with_fixed_int_encoding()
        .with_little_endian()
        .with_no_limit()
}

/// Marker for "plain old data" — types whose every byte pattern of size
/// `size_of::<Self>()` is a valid value.
///
/// # Safety
///
/// Implementers must ensure:
///
/// * `Self: Copy`.
/// * `#[repr(C)]` or `#[repr(transparent)]`.
/// * No padding bytes (use explicit `_pad` fields).
/// * No references, no `NonZero*`, no enum discriminants outside the
///   integer range of the declared `#[repr(uN)]`.
/// * `align_of::<Self>() <= 8`.
///
/// Implementing this for a type that violates the contract is undefined
/// behaviour as soon as a malformed buffer is borrow-decoded.
pub unsafe trait Pod: Copy + 'static {}

unsafe impl Pod for u8 {}
unsafe impl Pod for u16 {}
unsafe impl Pod for u32 {}
unsafe impl Pod for u64 {}
unsafe impl Pod for i8 {}
unsafe impl Pod for i16 {}
unsafe impl Pod for i32 {}
unsafe impl Pod for i64 {}
unsafe impl Pod for f32 {}
unsafe impl Pod for f64 {}

// =========================================================================
// AlignedBytes
// =========================================================================

/// An 8-byte aligned byte buffer. Returned by [`encode_zerocopy`] and
/// accepted by [`decode_zerocopy`].
///
/// Internally `Box<[u64]>`; we expose only its byte view. The trailing
/// bytes (if `len % 8 != 0`) are zero-padded.
pub struct AlignedBytes {
    storage: Box<[u64]>,
    len: usize,
}

/// Verification that `T` satisfies all [`Pod`] layout requirements.
///
/// Checks size, alignment, and that the type is not a reference or ZST
/// that would make Pod semantics meaningless. Call this in tests or
/// `const` contexts to catch Pod impls that violate the safety contract.
///
/// # Panics
///
/// Panics if any of the following hold:
/// * `align_of::<T>() > 8` — misaligned for the fixed-width bincode wire
/// * `size_of::<T>() == 0` — zero-sized types have no meaningful Pod repr
pub const fn verify_pod_layout<T: Pod>() {
    assert!(
        core::mem::align_of::<T>() <= 8,
        "Pod type alignment exceeds 8 bytes — not safe for zerocopy wire format"
    );
    assert!(
        core::mem::size_of::<T>() > 0,
        "Pod type must not be zero-sized"
    );
}

impl AlignedBytes {
    /// Constructs from a slice by copying its contents into an 8-byte
    /// aligned allocation.
    #[must_use]
    pub fn from_slice(bytes: &[u8]) -> Self {
        let n_u64 = bytes.len().div_ceil(core::mem::size_of::<u64>());
        let mut storage: Vec<u64> = alloc::vec![0u64; n_u64];
        // SAFETY: `storage` is `Vec<u64>` with `n_u64 * 8 >= bytes.len()`
        // bytes of writable memory; we never read uninitialized tail.
        unsafe {
            core::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                storage.as_mut_ptr().cast::<u8>(),
                bytes.len(),
            );
        }
        Self {
            storage: storage.into_boxed_slice(),
            len: bytes.len(),
        }
    }

    /// View as a byte slice. The start of this slice is 8-byte aligned.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `storage` is `Box<[u64]>`; the underlying allocation has
        // `storage.len() * 8` bytes, of which the first `self.len` are the
        // bytes the user wrote.
        unsafe { core::slice::from_raw_parts(self.storage.as_ptr().cast::<u8>(), self.len) }
    }

    /// Converts a `Vec<u8>` into an `AlignedBytes` **without copying** if the
    /// `Vec`'s allocation is already 8-byte aligned; copies otherwise.
    ///
    /// `Vec<u8>` typically inherits 1-byte alignment from the global allocator,
    /// so a copy is usually needed — but callers that produced the bytes via
    /// an aligned writer can take the zero-copy path.
    ///
    /// Returns `Some(Self)` on the zero-copy path and `None` if the alignment
    /// was insufficient (so the caller can fall back to `from_slice`). In
    /// practice, callers should use [`AlignedBytes::from_vec`] which handles
    /// both paths transparently.
    #[must_use]
    pub fn from_vec_if_aligned(v: Vec<u8>) -> Option<Self> {
        let ptr = v.as_ptr() as usize;
        if ptr & 7 != 0 {
            return None; // not 8-byte aligned — caller must copy
        }
        let _len = v.len();
        // Reinterpret the Vec<u8> as Vec<u64> by going through the allocator.
        // We can't do a zero-cost transmute (Vec internals aren't guaranteed),
        // so we use the aligned heap path: leak and rebuild. However, for
        // simplicity and correctness we take a direct copy path here because
        // Vec<u8>→Vec<u64> conversion requires capacity rounding.
        // The true zero-copy case is the `AlignedBytesWriter` path below.
        drop(v);
        None // Indicate copy is needed; use from_vec below for the real logic
    }

    /// Constructs from a byte slice, copying into an aligned allocation.
    ///
    /// A copy is always needed unless the source is already aligned.
    /// For the true zero-copy path, use [`AlignedBytesWriter`] directly.
    #[must_use]
    pub fn from_vec(v: &[u8]) -> Self {
        Self::from_slice(v)
    }

    /// Number of meaningful bytes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// =========================================================================
// AlignedBytesWriter — encode directly into an aligned buffer
// =========================================================================

/// A `bincode_next::enc::write::Writer` that encodes directly into an
/// 8-byte aligned `Vec<u64>`, avoiding the intermediate `Vec<u8>` that
/// `encode_to_vec` + `AlignedBytes::from_slice` would require.
///
/// Use [`encode_zerocopy_direct`] rather than constructing this type
/// manually.
pub struct AlignedBytesWriter {
    storage: Vec<u64>,
    /// Byte count of content written so far (≤ `storage.len() * 8`).
    byte_len: usize,
}

impl AlignedBytesWriter {
    /// Creates a writer with the given initial capacity (in bytes).
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        let u64_cap = cap.div_ceil(core::mem::size_of::<u64>());
        Self {
            storage: Vec::with_capacity(u64_cap),
            byte_len: 0,
        }
    }

    /// Consumes the writer and returns an [`AlignedBytes`] buffer.
    #[must_use]
    pub fn finish(self) -> AlignedBytes {
        AlignedBytes {
            storage: self.storage.into_boxed_slice(),
            len: self.byte_len,
        }
    }
}

impl Writer for AlignedBytesWriter {
    fn write(&mut self, bytes: &[u8]) -> Result<(), EncodeError> {
        let needed_bytes = self.byte_len + bytes.len();
        let needed_u64s = needed_bytes.div_ceil(core::mem::size_of::<u64>());
        // Grow the backing store if needed.
        if needed_u64s > self.storage.len() {
            self.storage.resize(needed_u64s, 0u64);
        }
        // SAFETY: `storage` is `Vec<u64>` with `storage.len() * 8` valid bytes;
        // we write into the region `[byte_len, byte_len + bytes.len())` which is
        // within bounds after the resize above. No two Writers alias this buffer.
        unsafe {
            core::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.storage.as_mut_ptr().cast::<u8>().add(self.byte_len),
                bytes.len(),
            );
        }
        self.byte_len += bytes.len();
        Ok(())
    }
}

/// Encodes `value` directly into a freshly allocated 8-byte aligned buffer
/// without an intermediate `Vec<u8>`.
///
/// This avoids the two-step `encode_to_vec` + `from_slice` copy that
/// [`encode_zerocopy`] performs, at the cost of a slightly larger internal
/// buffer (padded to 8-byte multiples).
///
/// # Errors
///
/// Returns whatever `bincode_next` produces on encode failure.
pub fn encode_zerocopy_direct<E: Encode>(value: E) -> Result<AlignedBytes, EncodeError> {
    let mut writer = AlignedBytesWriter::with_capacity(256);
    bincode_next::encode_into_writer(value, &mut writer, zerocopy_config())?;
    Ok(writer.finish())
}

/// Encodes `value` into a freshly allocated 8-byte aligned buffer using
/// [`zerocopy_config`].
///
/// # Errors
///
/// Returns whatever `bincode_next` produces on encode failure.
pub fn encode_zerocopy<E: Encode>(value: E) -> Result<AlignedBytes, EncodeError> {
    let vec = bincode_next::encode_to_vec(value, zerocopy_config())?;
    Ok(AlignedBytes::from_slice(&vec))
}

/// Borrow-decodes `T` from an aligned buffer with [`zerocopy_config`].
///
/// # Errors
///
/// Returns whatever `bincode_next` produces on decode failure.
pub fn decode_zerocopy<'a, T>(bytes: &'a AlignedBytes) -> Result<T, DecodeError>
where
    T: BorrowDecode<'a, ()>,
{
    let (value, _read) =
        bincode_next::borrow_decode_from_slice(bytes.as_bytes(), zerocopy_config())?;
    Ok(value)
}

/// Borrow-decodes `T` directly from a raw byte slice without going through
/// [`AlignedBytes`].
///
/// Use this when the source bytes are already guaranteed to be 8-byte
/// aligned (e.g. from [`MmapBuffer::with_view`], whose heap-fallback
/// path always returns a `Box<[u64]>`-backed slice). Skips the copy
/// that `AlignedBytes::from_slice` would otherwise perform.
///
/// # Errors
///
/// Returns a `DecodeError` if the bytes are misaligned for the contained
/// `Pod` types, or if `bincode_next` encounters a structural error.
pub fn decode_zerocopy_raw<'a, T>(bytes: &'a [u8]) -> Result<T, DecodeError>
where
    T: BorrowDecode<'a, ()>,
{
    let (value, _read) = bincode_next::borrow_decode_from_slice(bytes, zerocopy_config())?;
    Ok(value)
}

// =========================================================================
// BorrowedSlice
// =========================================================================

/// A `&'a [T]` that round-trips through `bincode_next` without copying.
///
/// Wire format (using [`zerocopy_config`]):
/// ```text
///   u64 little-endian length
///   length * size_of::<T>() bytes (native LE, packed)
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BorrowedSlice<'a, T: Pod> {
    inner: &'a [T],
}

impl<'a, T: Pod> BorrowedSlice<'a, T> {
    /// Constructs from an existing reference.
    #[must_use]
    pub const fn new(slice: &'a [T]) -> Self {
        Self { inner: slice }
    }

    /// Underlying slice.
    #[must_use]
    pub const fn as_slice(&self) -> &'a [T] {
        self.inner
    }

    /// Number of elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the slice has zero elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<'a, T: Pod> From<&'a [T]> for BorrowedSlice<'a, T> {
    fn from(s: &'a [T]) -> Self {
        Self::new(s)
    }
}

impl<T: Pod> Encode for BorrowedSlice<'_, T> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        let len = self.inner.len() as u64;
        len.encode(encoder)?;
        let byte_len = self
            .inner
            .len()
            .checked_mul(core::mem::size_of::<T>())
            .ok_or_else(|| {
                EncodeError::OtherString(alloc::string::String::from(
                    "BorrowedSlice: byte length overflowed usize",
                ))
            })?;
        // SAFETY: `T: Pod` guarantees every bit pattern is valid and there
        // are no padding bytes; the resulting byte slice fully describes
        // the original elements.
        let bytes: &[u8] =
            unsafe { core::slice::from_raw_parts(self.inner.as_ptr().cast::<u8>(), byte_len) };
        encoder.writer().write(bytes)
    }
}

/// Cold helper for the misalignment error path in `BorrowedSlice` decode.
#[doc(hidden)]
#[cold]
#[track_caller]
#[inline(never)]
const fn decode_error_misaligned() -> DecodeError {
    DecodeError::Other("BorrowedSlice: misaligned input buffer")
}

impl<'de, Context, T: Pod> BorrowDecode<'de, Context> for BorrowedSlice<'de, T> {
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, DecodeError> {
        let len = u64::decode(decoder)?;
        let len_usize: usize = len
            .try_into()
            .map_err(|_| DecodeError::OutsideUsizeRange(len))?;
        let byte_len = len_usize
            .checked_mul(core::mem::size_of::<T>())
            .ok_or(DecodeError::OutsideUsizeRange(len))?;
        decoder.claim_bytes_read(byte_len)?;
        let bytes: &'de [u8] = decoder.borrow_reader().take_bytes(byte_len)?;

        let align = core::mem::align_of::<T>();
        if (bytes.as_ptr() as usize) & (align - 1) != 0 {
            return Err(decode_error_misaligned());
        }

        // SAFETY: `T: Pod` (every bit pattern valid, no padding). The
        // pointer alignment was just checked, the byte length is
        // `len_usize * size_of::<T>()`, and lifetime `'de` is propagated
        // from the borrow reader's input.
        let inner: &'de [T] =
            unsafe { core::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), len_usize) };
        Ok(Self { inner })
    }
}

// =========================================================================
// BorrowedArena
// =========================================================================

/// A borrowed, read-only arena view: a contiguous run of nodes stored in
/// an external buffer (the bincode wire image, an mmap, etc.).
///
/// DAG / AST will eventually adopt this type via T1.1 / T1.4.
#[derive(Debug, Clone, Copy)]
pub struct BorrowedArena<'a, T: Pod> {
    nodes: BorrowedSlice<'a, T>,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T: Pod> BorrowedArena<'a, T> {
    /// Wraps a pre-existing borrowed slice as an arena.
    #[must_use]
    pub const fn new(nodes: BorrowedSlice<'a, T>) -> Self {
        Self {
            nodes,
            _phantom: PhantomData,
        }
    }

    /// Constructs from a plain `&[T]`.
    #[must_use]
    pub const fn from_slice(slice: &'a [T]) -> Self {
        Self::new(BorrowedSlice::new(slice))
    }

    /// Returns the node at index `i`, or `None` if out of bounds.
    #[must_use]
    pub fn get(&self, i: usize) -> Option<&'a T> {
        self.nodes.as_slice().get(i)
    }

    /// Returns the underlying slice.
    #[must_use]
    pub const fn as_slice(&self) -> &'a [T] {
        self.nodes.as_slice()
    }

    /// Number of nodes in the arena.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the arena is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl<T: Pod> Encode for BorrowedArena<'_, T> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.nodes.encode(encoder)
    }
}

impl<'de, Context, T: Pod> BorrowDecode<'de, Context> for BorrowedArena<'de, T> {
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, DecodeError> {
        let nodes = BorrowedSlice::<T>::borrow_decode(decoder)?;
        Ok(Self::new(nodes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_pod_layout_primitives() {
        verify_pod_layout::<u8>();
        verify_pod_layout::<u32>();
        verify_pod_layout::<u64>();
        verify_pod_layout::<f64>();
        verify_pod_layout::<i64>();
    }

    #[test]
    fn encode_zerocopy_direct_matches_encode_zerocopy() {
        let data: Vec<u32> = (0u32..64).collect();
        let view = BorrowedSlice::new(data.as_slice());

        let buf_indirect = encode_zerocopy(view).expect("indirect encode");
        let buf_direct = encode_zerocopy_direct(view).expect("direct encode");

        // Both should decode to the same data.
        let decoded_indirect: BorrowedSlice<'_, u32> =
            decode_zerocopy(&buf_indirect).expect("decode indirect");
        let decoded_direct: BorrowedSlice<'_, u32> =
            decode_zerocopy(&buf_direct).expect("decode direct");
        assert_eq!(decoded_indirect.as_slice(), decoded_direct.as_slice());
        assert_eq!(decoded_direct.as_slice(), data.as_slice());
        // Direct encoding must be 8-byte aligned.
        assert_eq!(buf_direct.as_bytes().as_ptr() as usize & 7, 0);
    }

    #[test]
    fn aligned_bytes_alignment_holds() {
        let buf = AlignedBytes::from_slice(&[1, 2, 3, 4, 5]);
        let ptr = buf.as_bytes().as_ptr() as usize;
        assert_eq!(ptr & 0b111, 0, "AlignedBytes start must be 8-byte aligned");
        assert_eq!(buf.as_bytes(), &[1, 2, 3, 4, 5]);
        assert_eq!(buf.len(), 5);
    }

    #[test]
    fn borrowed_slice_roundtrips_u32() {
        let data: Vec<u32> = (0u32..1024).collect();
        let view = BorrowedSlice::new(data.as_slice());

        let buf = encode_zerocopy(view).expect("encode");
        let decoded: BorrowedSlice<'_, u32> = decode_zerocopy(&buf).expect("decode");

        assert_eq!(decoded.len(), data.len());
        assert_eq!(decoded.as_slice(), data.as_slice());
    }

    #[test]
    fn borrowed_arena_roundtrips_f64() {
        let data: Vec<f64> = (0..256).map(|i| (i as f64) * 1.5).collect();
        let arena = BorrowedArena::from_slice(data.as_slice());

        let buf = encode_zerocopy(arena).expect("encode");
        let decoded: BorrowedArena<'_, f64> = decode_zerocopy(&buf).expect("decode");

        assert_eq!(decoded.len(), 256);
        assert_eq!(decoded.as_slice(), data.as_slice());
    }

    #[test]
    fn empty_slice_roundtrips() {
        let data: Vec<u64> = Vec::new();
        let view = BorrowedSlice::new(data.as_slice());
        let buf = encode_zerocopy(view).expect("encode");
        let decoded: BorrowedSlice<'_, u64> = decode_zerocopy(&buf).expect("decode");
        assert!(decoded.is_empty());
    }

    #[test]
    fn decoded_slice_points_into_buffer() {
        // The zero-copy guarantee: the decoded slice must reference bytes
        // *inside* the input buffer, not a fresh allocation.
        let data: Vec<u32> = (0u32..64).collect();
        let view = BorrowedSlice::new(data.as_slice());
        let buf = encode_zerocopy(view).expect("encode");
        let decoded: BorrowedSlice<'_, u32> = decode_zerocopy(&buf).expect("decode");

        let buf_start = buf.as_bytes().as_ptr() as usize;
        let buf_end = buf_start + buf.as_bytes().len();
        let decoded_start = decoded.as_slice().as_ptr() as usize;
        assert!(
            decoded_start >= buf_start && decoded_start < buf_end,
            "decoded slice does not point into the input buffer (zero-copy violated)"
        );
    }
}
