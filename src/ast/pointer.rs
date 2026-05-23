//! Relative pointer types for AST ↔ DAG referencing.
//!
//! `RelPtr` stores a signed index offset (typically `i32` or `i64`) between
//! elements in a contiguous buffer. This enables representing trees without
//! heap-allocated pointers, keeping layouts flat, cache-friendly, and safe.

use core::fmt;
use core::marker::PhantomData;



/// A relative pointer representing an offset in a contiguous buffer.
///
/// The offset type `O` is generic; in practice only `i32` (the default) is
/// used. Arena sizes are bounded to `u32::MAX` nodes, making a 4-byte offset
/// sufficient. The `i64` variant is kept generic but is not instantiated
/// anywhere in this library — remove or seal it if you need to minimize the
/// public API surface.
pub struct RelPtr<T, O = i32> {
    offset: O,
    _phantom: PhantomData<T>,
}

impl<T, O: Clone> Clone for RelPtr<T, O> {
    fn clone(&self) -> Self {
        Self {
            offset: self.offset.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T, O: Copy> Copy for RelPtr<T, O> {}

impl<T, O: PartialEq> PartialEq for RelPtr<T, O> {
    fn eq(&self, other: &Self) -> bool {
        self.offset.eq(&other.offset)
    }
}

impl<T, O: Eq> Eq for RelPtr<T, O> {}

impl<T, O: std::hash::Hash> std::hash::Hash for RelPtr<T, O> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.offset.hash(state);
    }
}

impl<T, O: bincode_next::enc::Encode> bincode_next::enc::Encode for RelPtr<T, O> {
    fn encode<E: bincode_next::enc::Encoder>(&self, encoder: &mut E) -> Result<(), bincode_next::error::EncodeError> {
        self.offset.encode(encoder)
    }
}

impl<T, O: bincode_next::de::Decode<C>, C> bincode_next::de::Decode<C> for RelPtr<T, O> {
    fn decode<D: bincode_next::de::Decoder<Context = C>>(decoder: &mut D) -> Result<Self, bincode_next::error::DecodeError> {
        let offset = O::decode(decoder)?;
        Ok(Self {
            offset,
            _phantom: PhantomData,
        })
    }
}

impl<'de, T, O: bincode_next::de::BorrowDecode<'de, C>, C> bincode_next::de::BorrowDecode<'de, C> for RelPtr<T, O> {
    fn borrow_decode<D: bincode_next::de::BorrowDecoder<'de, Context = C>>(decoder: &mut D) -> Result<Self, bincode_next::error::DecodeError> {
        let offset = O::borrow_decode(decoder)?;
        Ok(Self {
            offset,
            _phantom: PhantomData,
        })
    }
}

impl<T, O: fmt::Display> fmt::Debug for RelPtr<T, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RelPtr(offset: {})", self.offset)
    }
}

impl<T> RelPtr<T, i32> {
    /// The sentinel value used to represent a null (absent) relative pointer.
    ///
    /// `i32::MIN` is chosen because a valid offset of −2 147 483 648 would
    /// require the source and target indices to differ by exactly 2^31,
    /// which is impossible for any practical arena (bounded by `u32` ids).
    /// Using `0` as the sentinel (the previous choice) was incorrect: it
    /// collided with a valid pointer from any index to itself, and — more
    /// importantly — from any non-zero index to index 0 (the root).
    pub const NULL_OFFSET: i32 = i32::MIN;

    /// Creates a null relative pointer.
    #[must_use]
    pub const fn null() -> Self {
        Self {
            offset: Self::NULL_OFFSET,
            _phantom: PhantomData,
        }
    }

    /// Returns `true` if this relative pointer is null.
    #[must_use]
    pub const fn is_null(self) -> bool {
        self.offset == Self::NULL_OFFSET
    }

    /// Computes the relative pointer from a source index to a target index.
    ///
    /// When the offset overflows the `i32` range the function returns
    /// the null pointer (clamped fallback) rather than panicking.
    /// Callers that need to distinguish "real null" from "overflow"
    /// should use [`Self::from_indices_checked`].
    #[must_use]
    pub fn from_indices(source: usize, target: usize) -> Self {
        Self::from_indices_checked(source, target).unwrap_or_else(Self::null)
    }

    /// Like [`Self::from_indices`] but returns `None` on i32 overflow
    /// instead of clamping to null.
    #[must_use]
    pub fn from_indices_checked(source: usize, target: usize) -> Option<Self> {
        let diff = (target as isize) - (source as isize);
        let offset = i32::try_from(diff).ok()?;
        // Guard against accidentally encoding the null sentinel as a real
        // pointer (would require a 2^31-element distance — unreachable in
        // practice, but we make it explicit).
        if offset == Self::NULL_OFFSET {
            return None;
        }
        Some(Self {
            offset,
            _phantom: PhantomData,
        })
    }

    /// Resolves the target element's index in the slice, given the source index.
    ///
    /// Returns `None` if the relative pointer is null.
    #[must_use]
    pub fn resolve(self, source: usize) -> Option<usize> {
        if self.is_null() {
            return None;
        }
        let target = (source as isize) + (self.offset as isize);
        if target < 0 {
            None
        } else {
            Some(target as usize)
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relative_pointer_i32() {
        let source = 10;
        let target = 15;
        let ptr = RelPtr::<u64>::from_indices(source, target);
        assert!(!ptr.is_null());
        assert_eq!(ptr.resolve(source), Some(target));

        let backward_ptr = RelPtr::<u64>::from_indices(target, source);
        assert_eq!(backward_ptr.resolve(target), Some(source));

        let null_ptr = RelPtr::<u64>::null();
        assert!(null_ptr.is_null());
        assert_eq!(null_ptr.resolve(source), None);
    }
}
