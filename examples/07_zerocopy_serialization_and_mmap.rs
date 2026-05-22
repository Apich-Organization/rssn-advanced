//! Example 07: Zero-Copy Serialization and Memory-Mapped Arena
//!
//! This example demonstrates rssn-advanced's zero-allocation serialization path
//! using `BorrowedSlice` / `BorrowedArena` together with the `MmapBuffer`
//! file-backed storage. The key property shown is that `borrow_decode_from_slice`
//! returns a slice that points *directly into the source buffer* — no heap
//! allocation is made for the decoded element data.
//!
//! It also shows how to pair `AlignedBytes` with `encode_zerocopy` /
//! `decode_zerocopy` for in-memory round-trips, and how to verify the
//! zero-copy guarantee at runtime.
//!
//! Run with: `cargo run --example 07_zerocopy_serialization_and_mmap`

use std::fs;
use std::path::PathBuf;

use rssn_advanced::zerocopy::{
    AlignedBytes, BorrowedArena, BorrowedSlice, encode_zerocopy, decode_zerocopy, MmapBuffer,
};

fn main() {
    println!("=== RSSN-Advanced Example 07: Zero-Copy Serialization & Mmap ===\n");

    // -------------------------------------------------------------------------
    // 1. AlignedBytes: guaranteed 8-byte aligned allocation
    // -------------------------------------------------------------------------
    println!("Part 1: AlignedBytes alignment verification");
    let raw_bytes: Vec<u8> = (0u8..=15).collect();
    let aligned = AlignedBytes::from_slice(&raw_bytes);

    let ptr_addr = aligned.as_bytes().as_ptr() as usize;
    println!("  Input byte count               : {}", raw_bytes.len());
    println!("  AlignedBytes pointer address   : {:#x}", ptr_addr);
    println!("  Is 8-byte aligned?             : {}\n", ptr_addr % 8 == 0);

    // -------------------------------------------------------------------------
    // 2. BorrowedSlice: in-memory round-trip without heap allocation
    // -------------------------------------------------------------------------
    println!("Part 2: BorrowedSlice<u32> round-trip (encode → decode → verify zero-copy)");
    let data_u32: Vec<u32> = (100u32..164).collect(); // 64 elements
    let view = BorrowedSlice::new(data_u32.as_slice());

    let encoded_buf = encode_zerocopy(view).expect("encode failed");
    println!("  Source elements                : {}", data_u32.len());
    println!("  Wire format size (bytes)       : {}", encoded_buf.len());

    let decoded: BorrowedSlice<'_, u32> = decode_zerocopy(&encoded_buf).expect("decode failed");
    println!("  Decoded element count          : {}", decoded.len());

    // Zero-copy guarantee: decoded pointer must live inside encoded_buf bytes
    let buf_start = encoded_buf.as_bytes().as_ptr() as usize;
    let buf_end = buf_start + encoded_buf.as_bytes().len();
    let decoded_start = decoded.as_slice().as_ptr() as usize;
    let is_zero_copy = decoded_start >= buf_start && decoded_start < buf_end;
    println!("  Decoded slice inside source?   : {} (zero-copy: {})\n",
        if is_zero_copy { "YES" } else { "NO" },
        if is_zero_copy { "verified" } else { "violated!" }
    );

    assert!(is_zero_copy, "zero-copy invariant violated");

    // -------------------------------------------------------------------------
    // 3. BorrowedArena<f64>: encode an f64 arena, decode back, spot-check values
    // -------------------------------------------------------------------------
    println!("Part 3: BorrowedArena<f64> round-trip");
    let node_values: Vec<f64> = (0..128).map(|i| (i as f64) * std::f64::consts::PI).collect();
    let arena = BorrowedArena::from_slice(node_values.as_slice());

    let arena_buf = encode_zerocopy(arena).expect("arena encode failed");
    let decoded_arena: BorrowedArena<'_, f64> =
        decode_zerocopy(&arena_buf).expect("arena decode failed");

    println!("  Encoded node count             : {}", node_values.len());
    println!("  Decoded node count             : {}", decoded_arena.len());
    println!("  Node[0]  = {:.6} (expected {:.6})",
        decoded_arena.get(0).copied().unwrap_or(f64::NAN),
        0.0_f64
    );
    println!("  Node[1]  = {:.6} (expected {:.6})",
        decoded_arena.get(1).copied().unwrap_or(f64::NAN),
        std::f64::consts::PI
    );
    println!("  Node[63] = {:.6} (expected {:.6})\n",
        decoded_arena.get(63).copied().unwrap_or(f64::NAN),
        63.0 * std::f64::consts::PI
    );

    // -------------------------------------------------------------------------
    // 4. MmapBuffer: file-backed storage with zero-copy view
    // -------------------------------------------------------------------------
    println!("Part 4: MmapBuffer file-backed byte view");
    let tmp_path = PathBuf::from("./example_mmap_demo.bin");

    // Write encoded arena to a temporary file
    let bytes_to_write = encoded_buf.as_bytes();
    fs::write(&tmp_path, bytes_to_write).expect("failed to write temp file");
    println!("  Wrote {} bytes to {:?}", bytes_to_write.len(), tmp_path);

    // Open with MmapBuffer and decode without allocating
    let mmap = MmapBuffer::open(&tmp_path).expect("MmapBuffer::open failed");
    let mmap_len = mmap.with_view(|b| b.len());
    println!("  MmapBuffer byte length         : {}", mmap_len);

    // Verify the mmap bytes match the original
    let byte_match = mmap.with_view(|b| b == bytes_to_write);
    println!("  Mmap bytes match original?     : {}\n", byte_match);

    // Cleanup
    fs::remove_file(&tmp_path).unwrap_or_default();
    println!("  Temporary file removed.");

    println!("\n===================================================================");
}
