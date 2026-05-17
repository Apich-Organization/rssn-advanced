//! Disk-backed cache for DAG arena pages.
//!
//! Spills large DAG arenas to disk using high-performance serialization via bincode-next,
//! keeping RAM usage within bounds.

use std::fs::{create_dir_all, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use crate::dag::arena::DagArena;

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

    /// Spills the given `DagArena` to a file on disk.
    ///
    /// # Errors
    /// Returns an error if file creation or serialization fails.
    pub fn spill(&self, key: &str, arena: &DagArena) -> Result<(), String> {
        let filepath = self.cache_dir.join(format!("{key}.bin"));
        
        let config = bincode_next::config::standard();
        let bytes = bincode_next::encode_to_vec(arena, config)
            .map_err(|e| format!("Bincode serialization failed: {e:?}"))?;

        let mut file = File::create(&filepath)
            .map_err(|e| format!("Failed to create cache file {filepath:?}: {e:?}"))?;
        file.write_all(&bytes)
            .map_err(|e| format!("Failed to write cache bytes to file: {e:?}"))?;

        Ok(())
    }

    /// Restores a previously spilled `DagArena` from disk.
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened or deserialized.
    pub fn restore(&self, key: &str) -> Result<DagArena, String> {
        let filepath = self.cache_dir.join(format!("{key}.bin"));
        
        let mut file = File::open(&filepath)
            .map_err(|e| format!("Failed to open cache file {filepath:?}: {e:?}"))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| format!("Failed to read cache file: {e:?}"))?;

        let config = bincode_next::config::standard();
        let (arena, _): (DagArena, usize) = bincode_next::decode_from_slice(&bytes, config)
            .map_err(|e| format!("Bincode deserialization failed: {e:?}"))?;

        Ok(arena)
    }

    /// Deletes a spilled cache file from disk.
    ///
    /// # Errors
    /// Returns a message string if the file removal fails.
    pub fn delete(&self, key: &str) -> Result<(), String> {
        let filepath = self.cache_dir.join(format!("{key}.bin"));
        if filepath.exists() {
            std::fs::remove_file(&filepath)
                .map_err(|e| format!("Failed to delete cache file {filepath:?}: {e:?}"))?;
        }
        Ok(())
    }
}
