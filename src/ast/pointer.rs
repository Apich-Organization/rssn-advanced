//! Relative pointer types for AST ↔ DAG referencing.
//!
//! `RelPtr` stores a signed index offset (typically `i32` or `i64`) between
//! elements in a contiguous buffer. This enables representing trees without
//! heap-allocated pointers, keeping layouts flat, cache-friendly, and safe.

use core::fmt;
use core::marker::PhantomData;



/// A relative pointer representing an offset in a contiguous buffer.
///
/// It stores a relative offset of type `O` (typically `i32` or `i64`) from the
/// current element position to the target element position of type `T`.
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
    /// Creates a relative pointer pointing to a null target (represented by 0 offset).
    #[must_use]
    pub const fn null() -> Self {
        Self {
            offset: 0,
            _phantom: PhantomData,
        }
    }

    /// Returns `true` if this relative pointer is null.
    #[must_use]
    pub const fn is_null(self) -> bool {
        self.offset == 0
    }

    /// Computes the relative pointer from a source index to a target index.
    ///
    /// # Panics
    /// Panics if the offset overflow/underflow bounds of `i32`.
    #[must_use]
    pub fn from_indices(source: usize, target: usize) -> Self {
        if target == 0 {
            return Self::null();
        }
        let diff = (target as isize) - (source as isize);
        #[allow(clippy::cast_possible_truncation)]
        let offset = diff as i32;
        assert_eq!(offset as isize, diff, "Relative pointer offset overflowed i32");
        Self {
            offset,
            _phantom: PhantomData,
        }
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

impl<T> RelPtr<T, i64> {
    /// Creates a relative pointer pointing to a null target (represented by 0 offset).
    #[must_use]
    pub const fn null_i64() -> Self {
        Self {
            offset: 0,
            _phantom: PhantomData,
        }
    }

    /// Returns `true` if this relative pointer is null.
    #[must_use]
    pub const fn is_null_i64(self) -> bool {
        self.offset == 0
    }

    /// Computes the relative pointer from a source index to a target index.
    #[must_use]
    pub fn from_indices_i64(source: usize, target: usize) -> Self {
        if target == 0 {
            return Self::null_i64();
        }
        let offset = (target as isize) - (source as isize);
        #[allow(clippy::cast_possible_truncation)]
        let offset_val = offset as i64;
        Self {
            offset: offset_val,
            _phantom: PhantomData,
        }
    }

    /// Resolves the target element's index in the slice, given the source index.
    ///
    /// Returns `None` if the relative pointer is null.
    #[must_use]
    pub fn resolve_i64(self, source: usize) -> Option<usize> {
        if self.is_null_i64() {
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
