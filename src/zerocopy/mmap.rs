//! File-backed byte buffers for zero-copy decode.
//!
//! [`MmapBuffer`] presents a uniform interface over three backends:
//!
//! * **Linux + `libc` feature enabled** — `mmap(2)` with `MAP_PRIVATE`,
//!   page-aligned by definition.
//! * **Windows** — `CreateFileMappingW` + `MapViewOfFile`, also page-aligned.
//! * **Fallback (any platform)** — `std::fs::read` into a heap `Box<[u64]>`
//!   so the returned bytes start 8-byte aligned even without OS mmap.
//!
//! Callers receive bytes only through [`MmapBuffer::with_view`] so the
//! borrow cannot escape the mapping. This avoids the soundness traps of
//! self-referential types while still enabling true zero-copy decode.

#![allow(unsafe_code)]

use std::path::Path;

use crate::error::StorageError;

extern crate alloc;
use alloc::boxed::Box;
use alloc::vec::Vec;

// =========================================================================
// Public type
// =========================================================================

/// Owns a contiguous byte image of a file, exposed through a callback so
/// borrowed views cannot outlive the underlying mapping.
pub struct MmapBuffer {
    backing: Backing,
    len: usize,
}

impl MmapBuffer {
    /// Loads `path` using the best available backend for the platform.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Io`] if the file cannot be opened or read.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, StorageError> {
        #[cfg(all(target_os = "linux", feature = "libc"))]
        {
            if let Ok(mapped) = linux::map_private(path.as_ref()) {
                return Ok(mapped);
            }
            // Fall through to the portable fallback if mmap fails for any
            // reason (`/proc`-style pseudo files, NFS without mmap, ...).
        }

        #[cfg(target_os = "windows")]
        {
            if let Ok(mapped) = win::map_view(path.as_ref()) {
                return Ok(mapped);
            }
        }

        Self::read_to_aligned(path.as_ref())
    }

    /// Number of bytes in the mapping.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the mapping is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Runs `f` with a byte view of the mapping. The borrow may not escape
    /// `f`'s scope, so there is no opportunity for use-after-free.
    pub fn with_view<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        f(self.as_bytes())
    }

    fn as_bytes(&self) -> &[u8] {
        match &self.backing {
            Backing::Heap(b) => {
                // SAFETY: `b` is `Box<[u64]>` of `cap_u64`; we expose only
                // the first `self.len` bytes.
                unsafe { core::slice::from_raw_parts(b.as_ptr().cast::<u8>(), self.len) }
            }
            #[cfg(all(target_os = "linux", feature = "libc"))]
            Backing::Mmap(m) => {
                // SAFETY: `m.ptr` is a valid mmap region of `m.len` bytes
                // owned by this struct for as long as `&self` lives.
                unsafe { core::slice::from_raw_parts(m.ptr, self.len) }
            }
            #[cfg(target_os = "windows")]
            Backing::WinMap(m) => {
                // SAFETY: `m.ptr` is a valid `MapViewOfFile` region of
                // `m.len` bytes owned by this struct.
                unsafe { core::slice::from_raw_parts(m.ptr, self.len) }
            }
        }
    }

    fn read_to_aligned(path: &Path) -> Result<Self, StorageError> {
        let bytes: Vec<u8> = std::fs::read(path).map_err(|_| StorageError::Io)?;
        Self::from_bytes(&bytes)
    }

    /// Constructs an aligned in-memory buffer by copying `src`. Useful for
    /// tests and for round-tripping through
    /// [`crate::zerocopy::encode_zerocopy`].
    ///
    /// # Errors
    ///
    /// Currently infallible on platforms where heap allocation cannot
    /// signal failure; reserved for future backends.
    pub fn from_bytes(src: &[u8]) -> Result<Self, StorageError> {
        let n_u64 = src.len().div_ceil(core::mem::size_of::<u64>()).max(1);
        let mut storage: Vec<u64> = alloc::vec![0u64; n_u64];
        // SAFETY: writing `src.len()` bytes into a `Vec<u64>` allocation
        // that is at least `src.len()` bytes wide.
        unsafe {
            core::ptr::copy_nonoverlapping(
                src.as_ptr(),
                storage.as_mut_ptr().cast::<u8>(),
                src.len(),
            );
        }
        Ok(Self {
            backing: Backing::Heap(storage.into_boxed_slice()),
            len: src.len(),
        })
    }
}

// =========================================================================
// Backends
// =========================================================================

enum Backing {
    /// Portable fallback: bytes copied into an 8-byte aligned heap buffer.
    Heap(Box<[u64]>),
    #[cfg(all(target_os = "linux", feature = "libc"))]
    Mmap(linux::MmapRegion),
    #[cfg(target_os = "windows")]
    WinMap(win::WinMapping),
}

#[cfg(all(target_os = "linux", feature = "libc"))]
mod linux {
    use super::*;
    use core::ffi::c_void;
    use std::os::unix::io::AsRawFd;

    pub struct MmapRegion {
        pub ptr: *const u8,
        pub len: usize,
    }

    // SAFETY: the mmap region is read-only and lives for as long as the
    // owning `MmapBuffer`.
    unsafe impl Send for MmapRegion {}
    unsafe impl Sync for MmapRegion {}

    impl Drop for MmapRegion {
        fn drop(&mut self) {
            if !self.ptr.is_null() && self.len > 0 {
                // SAFETY: `munmap` takes a pointer/length pair we obtained
                // from `mmap` and never re-used elsewhere.
                unsafe {
                    libc::munmap(self.ptr as *mut c_void, self.len);
                }
            }
        }
    }

    pub fn map_private(path: &std::path::Path) -> Result<super::MmapBuffer, StorageError> {
        let file = std::fs::File::open(path).map_err(|_| StorageError::Io)?;
        let len = file.metadata().map_err(|_| StorageError::Io)?.len() as usize;
        if len == 0 {
            return super::MmapBuffer::from_bytes(&[]);
        }
        // SAFETY: standard mmap call with a real file descriptor. We
        // request PROT_READ + MAP_PRIVATE; the kernel returns a valid
        // pointer or MAP_FAILED.
        let ptr = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return crate::error::cold_storage_error_io();
        }
        Ok(super::MmapBuffer {
            backing: super::Backing::Mmap(MmapRegion {
                ptr: ptr.cast::<u8>(),
                len,
            }),
            len,
        })
    }
}

#[cfg(target_os = "windows")]
mod win {
    use super::*;
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::Security::SECURITY_ATTRIBUTES;
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_READ, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
        PAGE_READONLY, UnmapViewOfFile,
    };

    pub struct WinMapping {
        pub ptr: *const u8,
        view: MEMORY_MAPPED_VIEW_ADDRESS,
        mapping: HANDLE,
    }

    unsafe impl Send for WinMapping {}
    unsafe impl Sync for WinMapping {}

    impl Drop for WinMapping {
        fn drop(&mut self) {
            // SAFETY: `view` and `mapping` were produced by the matching
            // `MapViewOfFile` / `CreateFileMappingW` calls below.
            unsafe {
                UnmapViewOfFile(self.view);
                if !self.mapping.is_null() {
                    CloseHandle(self.mapping);
                }
            }
        }
    }

    pub fn map_view(path: &std::path::Path) -> Result<super::MmapBuffer, StorageError> {
        let file = std::fs::File::open(path).map_err(|_| StorageError::Io)?;
        let len = file.metadata().map_err(|_| StorageError::Io)?.len() as usize;
        if len == 0 {
            return super::MmapBuffer::from_bytes(&[]);
        }
        let raw_handle = file.as_raw_handle() as HANDLE;

        // SAFETY: standard CreateFileMappingW with a valid handle; the
        // kernel returns either a valid mapping handle or null.
        let mapping = unsafe {
            CreateFileMappingW(
                raw_handle,
                core::ptr::null::<SECURITY_ATTRIBUTES>(),
                PAGE_READONLY,
                0,
                0,
                core::ptr::null(),
            )
        };
        if mapping.is_null() {
            return crate::error::cold_storage_error_io();
        }

        // SAFETY: `mapping` was just successfully created; we ask for a
        // read-only view of the entire file.
        let view = unsafe { MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) };
        if view.Value.is_null() {
            unsafe {
                CloseHandle(mapping);
            }
            return crate::error::cold_storage_error_io();
        }

        Ok(super::MmapBuffer {
            backing: super::Backing::WinMap(WinMapping {
                ptr: view.Value.cast::<u8>(),
                view,
                mapping,
            }),
            len,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn read_fallback_returns_aligned_view() {
        let bytes: Vec<u8> = (0u8..=200).collect();
        let buf = MmapBuffer::from_bytes(&bytes).expect("alloc");
        let mut captured: Vec<u8> = Vec::new();
        buf.with_view(|view| {
            captured.extend_from_slice(view);
            // 8-byte alignment of the start.
            assert_eq!(view.as_ptr() as usize & 0b111, 0);
        });
        assert_eq!(captured, bytes);
        assert_eq!(buf.len(), bytes.len());
    }

    #[test]
    fn open_round_trip_via_tmpfile() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rssn_mmap_test_{}.bin", std::process::id()));
        {
            let mut f = std::fs::File::create(&path).expect("create");
            f.write_all(&[0xDE, 0xAD, 0xBE, 0xEF, 0x42, 0x00, 0x11, 0x22, 0x33])
                .expect("write");
        }
        let buf = MmapBuffer::open(&path).expect("open");
        buf.with_view(|view| {
            assert_eq!(view.len(), 9);
            assert_eq!(view[0], 0xDE);
            assert_eq!(view[8], 0x33);
            assert_eq!(view.as_ptr() as usize & 0b111, 0);
        });
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn empty_file_works() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rssn_mmap_empty_{}.bin", std::process::id()));
        std::fs::File::create(&path).expect("create");
        let buf = MmapBuffer::open(&path).expect("open");
        assert_eq!(buf.len(), 0);
        buf.with_view(|view| {
            assert!(view.is_empty());
        });
        let _ = std::fs::remove_file(&path);
    }
}
