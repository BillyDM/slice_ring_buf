# Slice Ring Buffer
![Test](https://github.com/BillyDM/slice_ring_buf/workflows/Test/badge.svg)
[![Documentation](https://docs.rs/slice_ring_buf/badge.svg)][documentation]
[![Crates.io](https://img.shields.io/crates/v/slice_ring_buf.svg)](https://crates.io/crates/slice_ring_buf)
[![License](https://img.shields.io/crates/l/slice_ring_buf.svg)](https://github.com/BillyDM/slice_ring_buf/blob/master/LICENSE)

A fast ring buffer implementation written in Rust. Copies/reads with slices are implemented with memcpy.

This is optimized for manipulating data in bulk with slices. The performance of indexing individual elements one at a time will be limited by the performance of the modulo (remainder) operation on an `isize` value. If your use case needs better indexing performance at the cost of possibly larger buffer sizes, and is not dependent on the buffer being an exact length, then take a look at my crate [`bit_mask_ring_buf`].

This crate has no consumer/producer logic, and is meant to be used as a raw data structure or a base for other data structures.

## Installation
Add `slice_ring_buf` as a dependency in your `Cargo.toml`:
```toml
slice_ring_buf = 0.1
```

## Example
```rust
use slice_ring_buf::{SliceRB, SliceRbRef};

// Create a ring buffer with type u32. The data will be
// initialized with the default value (0 in this case).
let mut rb = SliceRB::<u32>::from_len(4);
assert_eq!(rb.capacity(), 4);

// Memcpy data from a slice into the ring buffer at arbitrary
// `isize` indexes. Earlier data will not be copied if it will
// be overwritten by newer data, avoiding unecessary memcpy's.
// The correct placement of the newer data will still be preserved.
rb.write_latest(&[0, 2, 3, 4, 1], 0);
assert_eq!(rb[0], 1);
assert_eq!(rb[1], 2);
assert_eq!(rb[2], 3);
assert_eq!(rb[3], 4);

// Memcpy into slices at arbitrary `isize` indexes and length
let mut read_buffer = [0u32; 7];
rb.read_into(&mut read_buffer, 2);
assert_eq!(read_buffer, [3, 4, 1, 2, 3, 4, 1]);

// Read/write by retrieving slices directly
let (s1, s2) = rb.as_slices_len(1, 4);
assert_eq!(s1, &[2, 3, 4]);
assert_eq!(s2, &[1]);

// Read/write to buffer by indexing. Performance will be limited
// by the modulo (remainder) operation on an isize value.
rb[0] = 0;
rb[1] = 1;
rb[2] = 2;
rb[3] = 3;

// Wrap when reading/writing outside of bounds. Performance will be
// limited by the modulo (remainder) operation on an isize value.
assert_eq!(rb[-1], 3);
assert_eq!(rb[10], 2);

// Aligned/stack data may also be used
let mut stack_data = [0u32, 1, 2, 3];
let mut rb_ref = SliceRbRef::new(&mut stack_data);
rb_ref[-4] = 5;
let (s1, s2) = rb_ref.as_slices_len(0, 3);
assert_eq!(s1, &[5, 1, 2]);
assert_eq!(s2, &[]);
```

[documentation]: https://docs.rs/slice_ring_buf/
[`bit_mask_ring_buf`]: https://crates.io/crates/bit_mask_ring_buf/