//! Disk-backed cache for DAG arena pages.
//!
//! Per `storage_review §1` the previous implementation called
//! `file.read_to_end(&mut bytes)` — loading multi-GB arenas wholesale
//! into a fresh `Vec<u8>` before decoding. This rewrite:
//!
//! * Encodes through [`PackedArenaImage`] so the on-disk image is the
//!   32-byte-per-node packed wire form (`plan.md §2.2`).
//! * Restores by opening the file with [`MmapBuffer`] (Linux/Windows
//!   mmap or aligned-heap fallback) and decoding via
//!   [`BorrowedArenaView`] — no `read_to_end`, no whole-file
//!   intermediate allocation.
//! * `load_borrowed` exposes the zero-copy [`BorrowedArenaView`]
//!   directly via a callback so the borrow can't escape the mapping.

use std::fs::{File, create_dir_all};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use crate::dag::arena::DagArena;
use crate::dag::packed::{BorrowedArenaView, PackedArenaImage};
use crate::zerocopy::MmapBuffer;

/// Disk-backed spillover cache for archiving/loading large DAG arenas.
#[derive(Debug, Clone)]
pub struct DiskCache {
    cache_dir: PathBuf,
}

impl DiskCache {
    /// Creates a new `DiskCache` with the given backing directory.
    ///
    /// # Errors
    /// Returns a `std::io::Error` if directory creation fails.
    pub fn new<P: AsRef<Path>>(cache_dir: P) -> std::io::Result<Self> {
        let path = cache_dir.as_ref().to_path_buf();
        create_dir_all(&path)?;
        Ok(Self { cache_dir: path })
    }

    fn key_path(&self, key: &str) -> PathBuf {
        self.cache_dir.join(format!("{key}.bin"))
    }

    /// Spills the given `DagArena` to a file on disk.
    ///
    /// Packs the arena to the 32-byte-per-node wire form, then streams
    /// the encoded bytes directly into a `BufWriter<File>` — avoiding
    /// the intermediate `AlignedBytes` copy that the old
    /// `encode()` + `write_all` path required.
    ///
    /// # Errors
    /// Returns an error if file creation, packing, or write fails.
    pub fn spill(&self, key: &str, arena: &DagArena) -> Result<(), String> {
        let image = PackedArenaImage::from_arena(arena);
        let filepath = self.key_path(key);
        let file = File::create(&filepath)
            .map_err(|e| format!("Failed to create cache file {}: {e:?}", filepath.display()))?;
        let buf_writer = BufWriter::new(file);
        image
            .write_to(buf_writer)
            .map_err(|e| format!("Failed to stream arena to file: {e:?}"))
    }

    /// Restores a previously spilled `DagArena` from disk.
    ///
    /// Opens the file via [`MmapBuffer`] (so no `read_to_end`-style
    /// whole-file `Vec<u8>` allocation), borrow-decodes the packed
    /// image, and materializes it back to an owned `DagArena`.
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened or decoded.
    pub fn restore(&self, key: &str) -> Result<DagArena, String> {
        let filepath = self.key_path(key);
        let mm = MmapBuffer::open(&filepath)
            .map_err(|e| format!("Failed to open cache file {}: {e:?}", filepath.display()))?;
        // `MmapBuffer` guarantees 8-byte alignment on every backend
        // (heap: Box<[u64]>; mmap: page-aligned). We decode directly
        // from the raw bytes without a redundant AlignedBytes copy.
        let arena_owned = mm.with_view(|bytes| -> Result<DagArena, String> {
            let view = crate::zerocopy::decode_zerocopy_raw::<BorrowedArenaView<'_>>(bytes)
                .map_err(|e| format!("Packed arena decode failed: {e:?}"))?;
            Ok(view.to_owned_arena())
        })?;
        Ok(arena_owned)
    }

    /// Opens the spilled file via mmap and hands the zero-copy
    /// [`BorrowedArenaView`] to `f`. The view's lifetime is bounded by
    /// the closure, so it cannot escape the mapping.
    ///
    /// Prefer this over [`Self::restore`] for read-only workloads —
    /// it never allocates a fresh `DagArena`.
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened or decoded.
    pub fn load_borrowed<R>(
        &self,
        key: &str,
        f: impl FnOnce(BorrowedArenaView<'_>) -> R,
    ) -> Result<R, String> {
        let filepath = self.key_path(key);
        let mm = MmapBuffer::open(&filepath)
            .map_err(|e| format!("Failed to open cache file {}: {e:?}", filepath.display()))?;
        let out = mm.with_view(|bytes| -> Result<R, String> {
            let view = crate::zerocopy::decode_zerocopy_raw::<BorrowedArenaView<'_>>(bytes)
                .map_err(|e| format!("Packed arena decode failed: {e:?}"))?;
            Ok(f(view))
        })?;
        Ok(out)
    }

    /// Deletes a spilled cache file from disk.
    ///
    /// # Errors
    /// Returns a message string if the file removal fails.
    pub fn delete(&self, key: &str) -> Result<(), String> {
        let filepath = self.key_path(key);
        if filepath.exists() {
            std::fs::remove_file(&filepath)
                .map_err(|e| format!("Failed to delete cache file {}: {e:?}", filepath.display()))?;
        }
        Ok(())
    }
}

// =========================================================================
// LazyDiskArena — deferred materialization of a spilled arena
// =========================================================================

/// A lazy handle to a spilled arena file.
///
/// Opening a spilled cache file and decoding it into a `DagArena` eagerly
/// pays the full allocation cost even for read-only queries that only
/// inspect a handful of nodes. `LazyDiskArena` defers that cost:
///
/// * `open` stores only the file path.
/// * `with_view` maps the file, borrow-decodes the packed image, and calls
///   the user closure — no allocation beyond the mmap itself.
/// * `load` materialises an owned `DagArena` from the mmap when the caller
///   needs a fully writable copy.
#[derive(Debug, Clone)]
pub struct LazyDiskArena {
    path: PathBuf,
}

impl LazyDiskArena {
    /// Opens a `LazyDiskArena` pointing at `path`.
    ///
    /// Does not open or stat the file — defers everything to the first
    /// [`Self::with_view`] / [`Self::load`] call.
    #[must_use]
    pub fn open<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    /// Maps the file and hands the zero-copy [`BorrowedArenaView`] to `f`.
    ///
    /// The view's lifetime is bounded to the closure, so it cannot outlive
    /// the mapping. The mapping is released when `f` returns.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or decoded.
    pub fn with_view<R>(
        &self,
        f: impl FnOnce(BorrowedArenaView<'_>) -> R,
    ) -> Result<R, String> {
        let mm = MmapBuffer::open(&self.path)
            .map_err(|e| format!("LazyDiskArena: cannot open {}: {e:?}", self.path.display()))?;
        let out = mm.with_view(|bytes| -> Result<R, String> {
            let view = crate::zerocopy::decode_zerocopy_raw::<BorrowedArenaView<'_>>(bytes)
                .map_err(|e| format!("LazyDiskArena: decode failed: {e:?}"))?;
            Ok(f(view))
        })?;
        Ok(out)
    }

    /// Materialises a fully owned `DagArena` from the spilled file.
    ///
    /// Equivalent to [`DiskCache::restore`] but without needing a `DiskCache`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or decoded.
    pub fn load(&self) -> Result<DagArena, String> {
        self.with_view(|view| view.to_owned_arena())
    }

    /// Path to the backing file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::builder::DagBuilder;

    fn tmp_subdir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("rssn_diskcache_{}_{}", std::process::id(), name))
    }

    #[test]
    fn round_trip_arena_via_mmap() {
        let dir = tmp_subdir("rt");
        let cache = DiskCache::new(&dir).expect("mkdir");
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let y = b.variable("y");
        let sum = b.add(x, y);
        let c = b.constant(3.5);
        let _ = b.mul(sum, c);

        let original_len = b.arena().len();
        cache.spill("k", b.arena()).expect("spill");
        let restored = cache.restore("k").expect("restore");
        assert_eq!(restored.len(), original_len);
        // Every node has matching kind/value/arity after round-trip.
        for i in 0..original_len {
            let orig = b
                .arena()
                .get(crate::dag::node::DagNodeId::new(i as u32))
                .expect("orig");
            let new = restored
                .get(crate::dag::node::DagNodeId::new(i as u32))
                .expect("new");
            assert_eq!(orig.kind, new.kind);
            assert_eq!(orig.value, new.value);
            assert_eq!(orig.meta.arity, new.meta.arity);
            assert_eq!(orig.children.as_slice(), new.children.as_slice());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_borrowed_returns_view_without_copying_nodes() {
        let dir = tmp_subdir("borrowed");
        let cache = DiskCache::new(&dir).expect("mkdir");
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let y = b.variable("y");
        let _ = b.add(x, y);

        cache.spill("k", b.arena()).expect("spill");
        let len = cache
            .load_borrowed("k", |view| view.len())
            .expect("load");
        assert_eq!(len, b.arena().len());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lazy_disk_arena_with_view_and_load() {
        let dir = tmp_subdir("lazy");
        let cache = DiskCache::new(&dir).expect("mkdir");
        let mut b = DagBuilder::new();
        let x = b.variable("x");
        let y = b.variable("y");
        let _ = b.add(x, y);

        cache.spill("k", b.arena()).expect("spill");
        let path = dir.join("k.bin");
        let lazy = LazyDiskArena::open(&path);

        // with_view: check node count without full materialisation
        let len = lazy.with_view(|v| v.len()).expect("with_view");
        assert_eq!(len, b.arena().len());

        // load: full arena materialisation
        let arena = lazy.load().expect("load");
        assert_eq!(arena.len(), b.arena().len());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_removes_file() {
        let dir = tmp_subdir("del");
        let cache = DiskCache::new(&dir).expect("mkdir");
        let mut b = DagBuilder::new();
        let _ = b.variable("x");
        cache.spill("k", b.arena()).expect("spill");
        assert!(dir.join("k.bin").exists());
        cache.delete("k").expect("delete");
        assert!(!dir.join("k.bin").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
