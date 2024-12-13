use alloc::vec::Vec;
use core::{fmt::Debug, num::NonZeroUsize};

use crate::inner;

/// A ring buffer implementation optimized for working with slices. Note this pretty
/// much does the same thing as [`VecDeque`], but with the added ability to index
/// using negative values, as well as working with buffers allocated on the stack.
/// This struct can be used without the standard library (`#![no_std]`).
///
/// This struct has no consumer/producer logic, and is meant to be used for DSP or as
/// a base for other data structures.
///
/// This data type is optimized for manipulating data in chunks with slices.
/// Indexing one element at a time is slow.
///
/// The length of this ring buffer cannot be `0`.
/// 
/// ## Example
/// ```rust
/// # use core::num::NonZeroUsize;
/// # use slice_ring_buf::SliceRB;
/// // Create a ring buffer with type u32. The data will be
/// // initialized with the value of `0`.
/// let mut rb = SliceRB::<u32>::new(NonZeroUsize::new(4).unwrap(), 0);
///
/// // Memcpy data from a slice into the ring buffer at arbitrary
/// // `isize` indexes. Earlier data will not be copied if it will
/// // be overwritten by newer data, avoiding unecessary memcpy's.
/// // The correct placement of the newer data will still be preserved.
/// rb.write_latest(&[0, 2, 3, 4, 1], 0);
/// assert_eq!(rb[0], 1);
/// assert_eq!(rb[1], 2);
/// assert_eq!(rb[2], 3);
/// assert_eq!(rb[3], 4);
///
/// // Memcpy into slices at arbitrary `isize` indexes and length.
/// let mut read_buffer = [0u32; 7];
/// rb.read_into(&mut read_buffer, 2);
/// assert_eq!(read_buffer, [3, 4, 1, 2, 3, 4, 1]);
///
/// // Read/write by retrieving slices directly.
/// let (s1, s2) = rb.as_slices_len(1, 4);
/// assert_eq!(s1, &[2, 3, 4]);
/// assert_eq!(s2, &[1]);
///
/// // Read/write to buffer by indexing. (Note that indexing
/// // one element at a time is slow.)
/// rb[0] = 0;
/// rb[1] = 1;
/// rb[2] = 2;
/// rb[3] = 3;
///
/// // Wrap when reading/writing outside of bounds.
/// assert_eq!(rb[-1], 3);
/// assert_eq!(rb[10], 2);
/// ```
///
/// [`VecDeque`]: https://doc.rust-lang.org/std/collections/struct.VecDeque.html
pub struct SliceRB<T> {
    vec: Vec<T>,
}

impl<T> SliceRB<T> {
    /// Creates a new [`SliceRB`] with the given vec as its data source.
    ///
    /// # Panics
    ///
    /// Panics if `vec.len()` is equal to `0` or  is greater than `isize::MAX`.
    /// 
    /// # Example
    /// ```
    /// # use slice_ring_buf::SliceRB;
    /// let rb = SliceRB::<u32>::from_vec(vec![0, 1, 2, 3]);
    ///
    /// assert_eq!(rb.len().get(), 4);
    /// assert_eq!(rb[-3], 1);
    /// ```
    pub fn from_vec(vec: Vec<T>) -> Self {
        assert!(!vec.is_empty());
        assert!(vec.len() <= isize::MAX as usize);

        Self { vec }
    }

    /// Creates a new [`SliceRB`] without initializing data.
    ///
    /// * `len` - The length of the ring buffer.
    ///
    /// # Safety
    ///
    /// * Undefined behavior may occur if uninitialized data is read from. By using
    /// this you assume the responsibility of making sure any data is initialized
    /// before it is read.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// unsafe {
    ///     let rb = SliceRB::<u32>::new_uninit(NonZeroUsize::new(3).unwrap());
    ///     assert_eq!(rb.len().get(), 3);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if `len > isize::MAX`.
    /// * This will panic if allocation fails due to being out of memory.
    pub unsafe fn new_uninit(len: NonZeroUsize) -> Self {
        assert!(len.get() <= isize::MAX as usize);

        let mut vec = Vec::with_capacity(len.get());
        vec.set_len(len.get());

        Self { vec }
    }

    /// Creates a new [`SliceRB`] with an allocated capacity equal to exactly the
    /// given length. No data will be initialized.
    ///
    /// * `len` - The length of the ring buffer.
    ///
    /// # Safety
    ///
    /// * Undefined behavior may occur if uninitialized data is read from. By using
    /// this you assume the responsibility of making sure any data is initialized
    /// before it is read.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// unsafe {
    ///     let rb = SliceRB::<u32>::new_exact_uninit(NonZeroUsize::new(3).unwrap());
    ///     assert_eq!(rb.len().get(), 3);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if `len > isize::MAX`.
    /// * This will panic if allocation fails due to being out of memory.
    pub unsafe fn new_exact_uninit(len: NonZeroUsize) -> Self {
        assert!(len.get() <= isize::MAX as usize);

        let mut vec = Vec::new();
        vec.reserve_exact(len.get());
        vec.set_len(len.get());

        Self { vec }
    }

    /// Creates a new [`SliceRB`] without initializing data, while reserving extra
    /// capacity for future changes to `len`.
    ///
    /// * `len` - The length of the ring buffer.
    /// * `capacity` - The allocated capacity of the ring buffer. If this is less than
    /// `len`, then it will be ignored.
    ///
    /// # Safety
    ///
    /// * Undefined behavior may occur if uninitialized data is read from. By using
    /// this you assume the responsibility of making sure any data is initialized
    /// before it is read.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// unsafe {
    ///     let rb = SliceRB::<u32>::with_capacity_uninit(NonZeroUsize::new(3).unwrap(), 10);
    ///     assert_eq!(rb.len().get(), 3);
    ///     assert!(rb.capacity().get() >= 10);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if `len > isize::MAX` or `capacity > isize::MAX`.
    /// * This will panic if allocation fails due to being out of memory.
    pub unsafe fn with_capacity_uninit(len: NonZeroUsize, capacity: usize) -> Self {
        assert!(len.get() <= isize::MAX as usize);
        assert!(capacity <= isize::MAX as usize);

        let mut vec = Vec::with_capacity(core::cmp::max(len.get(), capacity));
        vec.set_len(len.get());

        Self { vec }
    }

    /// Returns the length of the ring buffer.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let rb = SliceRB::<u32>::new(NonZeroUsize::new(4).unwrap(), 0);
    ///
    /// assert_eq!(rb.len().get(), 4);
    /// ```
    pub fn len(&self) -> NonZeroUsize {
        // SAFETY:
        // * All constructors and other methods which modify the length ensure
        // that the length is greater than `0`.
        unsafe { NonZeroUsize::new_unchecked(self.vec.len()) }
    }

    /// Returns the allocated capacity of the internal vector.
    ///
    /// Please note this is not the same as the length of the buffer.
    /// For that use `SliceRB::len()`.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let rb = SliceRB::<u32>::new(NonZeroUsize::new(4).unwrap(), 0);
    ///
    /// assert!(rb.capacity().get() >= 4);
    /// ```
    pub fn capacity(&self) -> NonZeroUsize {
        // SAFETY:
        // * All constructors and other methods which modify the length ensure
        // that the length is greater than `0`.
        unsafe { NonZeroUsize::new_unchecked(self.vec.capacity()) }
    }

    /// Returns the actual index of the ring buffer from the given
    /// `i` index.
    ///
    /// * First, a bounds check will be performed. If it is within bounds,
    /// then it is simply returned.
    /// * If it is not in bounds, then performance will
    /// be limited by the modulo (remainder) operation on an `isize` value.
    ///
    /// # Performance
    ///
    /// Prefer to manipulate data in bulk with methods that return slices. If you
    /// need to index multiple elements one at a time, prefer to use
    /// `SliceRB::at(&mut i)` over `SliceRB[i]` to reduce the number of
    /// modulo operations to perform.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let rb = SliceRB::<u32>::new(NonZeroUsize::new(4).unwrap(), 0);
    ///
    /// assert_eq!(rb.constrain(2), 2);
    /// assert_eq!(rb.constrain(4), 0);
    /// assert_eq!(rb.constrain(-3), 1);
    /// assert_eq!(rb.constrain(7), 3);
    /// ```
    pub fn constrain(&self, i: isize) -> isize {
        inner::constrain(i, self.vec.len() as isize)
    }

    /// Sets the length of the ring buffer without initializing any newly allocated data.
    ///
    /// * If `len` is less than the current length, then the data
    /// will be truncated.
    /// * If `len` is larger than the current length, then all newly
    /// allocated elements appended to the end will be unitialized.
    ///
    /// # Safety
    ///
    /// * Undefined behavior may occur if uninitialized data is read from. By using
    /// this you assume the responsibility of making sure any data is initialized
    /// before it is read.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::new(NonZeroUsize::new(2).unwrap(), 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    ///
    /// unsafe {
    ///     rb.set_len_uninit(NonZeroUsize::new(4).unwrap());
    ///
    ///     assert_eq!(rb.len().get(), 4);
    ///
    ///     assert_eq!(rb[0], 1);
    ///     assert_eq!(rb[1], 2);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if `len > isize::MAX`.
    /// * This will panic if allocation fails due to being out of memory.
    pub unsafe fn set_len_uninit(&mut self, len: NonZeroUsize) {
        assert!(len.get() <= isize::MAX as usize);

        if len.get() != self.vec.len() {
            if len.get() > self.vec.len() {
                // Extend without initializing.
                self.vec.reserve(len.get() - self.vec.len());
            }
            self.vec.set_len(len.get());
        }
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the internal `Vec`. This is equivalant to `Vec::reserve()`.
    ///
    /// The collection may reserve more space to avoid frequent reallocations. After
    /// calling reserve, capacity will be greater than or equal to self.len() + additional.
    /// Does nothing if capacity is already sufficient.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::new(NonZeroUsize::new(2).unwrap(), 0);
    ///
    /// rb.reserve(8);
    ///
    /// assert!(rb.capacity().get() >= 10);
    /// ```
    ///
    /// # Panics
    ///
    /// * Panics if out of memory.
    pub fn reserve(&mut self, additional: usize) {
        self.vec.reserve(additional);
    }

    /// Reserves capacity for exactly `additional` more elements to be inserted
    /// in the internal `Vec`. This is equivalant to `Vec::reserve_exact()`.
    ///
    /// The collection may reserve more space to avoid frequent reallocations. After
    /// calling reserve, capacity will be greater than or equal to self.len() + additional.
    /// Does nothing if capacity is already sufficient.
    ///
    /// Note that the allocator may give the collection more space than it requests. Therefore,
    /// capacity can not be relied upon to be precisely minimal. Prefer `reserve` if future
    /// insertions are expected.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::new(NonZeroUsize::new(2).unwrap(), 0);
    ///
    /// rb.reserve_exact(8);
    ///
    /// assert!(rb.capacity().get() >= 10);
    /// ```
    ///
    /// # Panics
    ///
    /// * Panics if out of memory.
    pub fn reserve_exact(&mut self, additional: usize) {
        self.vec.reserve_exact(additional);
    }

    /// Shrinks the capacity of the internal `Vec` as much as possible. This is equivalant to
    /// `Vec::shrink_to_fit`.
    ///
    /// It will drop down as close as possible to the length but the allocator may still inform
    /// the vector that there is space for a few more elements.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::new(NonZeroUsize::new(2).unwrap(), 0);
    ///
    /// rb.reserve(8);
    /// assert!(rb.capacity().get() >= 10);
    ///
    /// rb.shrink_to_fit();
    /// assert!(rb.capacity().get() >= 2);
    /// ```
    pub fn shrink_to_fit(&mut self) {
        self.vec.shrink_to_fit();
    }

    /// Returns two slices that contain all the data in the ring buffer
    /// starting at the index `start`.
    ///
    /// # Returns
    ///
    /// * The first slice is the starting chunk of data. This will never be empty.
    /// * The second slice is the second contiguous chunk of data. This may
    /// or may not be empty depending if the buffer needed to wrap around to the beginning of
    /// its internal memory layout.
    ///
    /// # Performance
    ///
    /// Prefer to use this to manipulate data in bulk over indexing one element at a time.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::new(NonZeroUsize::new(4).unwrap(), 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// let (s1, s2) = rb.as_slices(-4);
    /// assert_eq!(s1, &[1, 2, 3, 4]);
    /// assert_eq!(s2, &[]);
    ///
    /// let (s1, s2) = rb.as_slices(3);
    /// assert_eq!(s1, &[4]);
    /// assert_eq!(s2, &[1, 2, 3]);
    /// ```
    pub fn as_slices(&self, start: isize) -> (&[T], &[T]) {
        // SAFETY:
        // * All constructors ensure that the length of `vec` is greater than `0`.
        // * All constructors and other methods which modify the length ensure
        // that the length is less than or equal to `isize::MAX`.
        unsafe { inner::as_slices(start, &self.vec) }
    }

    /// Returns two slices of data in the ring buffer
    /// starting at the index `start` and with length `len`.
    ///
    /// * `start` - The starting index
    /// * `len` - The length of data to read. If `len` is greater than the
    /// length of the ring buffer, then that length will be used instead.
    ///
    /// # Returns
    ///
    /// * The first slice is the starting chunk of data.
    /// * The second slice is the second contiguous chunk of data. This may
    /// or may not be empty depending if the buffer needed to wrap around to the beginning of
    /// its internal memory layout.
    ///
    /// # Performance
    ///
    /// Prefer to use this to manipulate data in bulk over indexing one element at a time.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::new(NonZeroUsize::new(4).unwrap(), 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// let (s1, s2) = rb.as_slices_len(-4, 3);
    /// assert_eq!(s1, &[1, 2, 3]);
    /// assert_eq!(s2, &[]);
    ///
    /// let (s1, s2) = rb.as_slices_len(3, 5);
    /// assert_eq!(s1, &[4]);
    /// assert_eq!(s2, &[1, 2, 3]);
    /// ```
    pub fn as_slices_len(&self, start: isize, len: usize) -> (&[T], &[T]) {
        // SAFETY:
        // * All constructors ensure that the length of `vec` is greater than `0`.
        // * All constructors and other methods which modify the length ensure
        // that the length is less than or equal to `isize::MAX`.
        unsafe { inner::as_slices_len(start, len, &self.vec) }
    }

    /// Returns two slices of data in the ring buffer
    /// starting at the index `start` and with length `len`. If `len` is greater
    /// than the length of the ring buffer, then the buffer's length will be used
    /// instead, while still preserving the position of the last element.
    ///
    /// * `start` - The starting index
    /// * `len` - The length of data to read. If `len` is greater than the
    /// length of the ring buffer, then the buffer's length will be used instead, while
    /// still preserving the position of the last element.
    ///
    /// # Returns
    ///
    /// * The first slice is the starting chunk of data.
    /// * The second slice is the second contiguous chunk of data. This may
    /// or may not be empty depending if the buffer needed to wrap around to the beginning of
    /// its internal memory layout.
    ///
    /// # Performance
    ///
    /// Prefer to use this to manipulate data in bulk over indexing one element at a time.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::new(NonZeroUsize::new(4).unwrap(), 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// let (s1, s2) = rb.as_slices_latest(-4, 3);
    /// assert_eq!(s1, &[1, 2, 3]);
    /// assert_eq!(s2, &[]);
    ///
    /// let (s1, s2) = rb.as_slices_latest(0, 5);
    /// assert_eq!(s1, &[2, 3, 4]);
    /// assert_eq!(s2, &[1]);
    /// ```
    pub fn as_slices_latest(&self, start: isize, len: usize) -> (&[T], &[T]) {
        // SAFETY:
        // * All constructors ensure that the length of `vec` is greater than `0`.
        // * All constructors and other methods which modify the length ensure
        // that the length is less than or equal to `isize::MAX`.
        unsafe { inner::as_slices_latest(start, len, &self.vec) }
    }

    /// Returns two mutable slices that contain all the data in the ring buffer
    /// starting at the index `start`.
    ///
    /// # Returns
    ///
    /// * The first slice is the starting chunk of data. This will never be empty.
    /// * The second slice is the second contiguous chunk of data. This may
    /// or may not be empty depending if the buffer needed to wrap around to the beginning of
    /// its internal memory layout.
    ///
    /// # Performance
    ///
    /// Prefer to use this to manipulate data in bulk over indexing one element at a time.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::new(NonZeroUsize::new(4).unwrap(), 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// let (s1, s2) = rb.as_mut_slices(-4);
    /// assert_eq!(s1, &mut [1, 2, 3, 4]);
    /// assert_eq!(s2, &mut []);
    ///
    /// let (s1, s2) = rb.as_mut_slices(3);
    /// assert_eq!(s1, &mut [4]);
    /// assert_eq!(s2, &mut [1, 2, 3]);
    /// ```
    pub fn as_mut_slices(&mut self, start: isize) -> (&mut [T], &mut [T]) {
        // SAFETY:
        // * All constructors ensure that the length of `vec` is greater than `0`.
        // * All constructors and other methods which modify the length ensure
        // that the length is less than or equal to `isize::MAX`.
        unsafe { inner::as_mut_slices(start, &mut self.vec) }
    }

    /// Returns two mutable slices of data in the ring buffer
    /// starting at the index `start` and with length `len`.
    ///
    /// * `start` - The starting index
    /// * `len` - The length of data to read. If `len` is greater than the
    /// length of the ring buffer, then that length will be used instead.
    ///
    /// # Returns
    ///
    /// * The first slice is the starting chunk of data.
    /// * The second slice is the second contiguous chunk of data. This may
    /// or may not be empty depending if the buffer needed to wrap around to the beginning of
    /// its internal memory layout.
    ///
    /// # Performance
    ///
    /// Prefer to use this to manipulate data in bulk over indexing one element at a time.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::new(NonZeroUsize::new(4).unwrap(), 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// let (s1, s2) = rb.as_mut_slices_len(-4, 3);
    /// assert_eq!(s1, &mut [1, 2, 3]);
    /// assert_eq!(s2, &mut []);
    ///
    /// let (s1, s2) = rb.as_mut_slices_len(3, 5);
    /// assert_eq!(s1, &mut [4]);
    /// assert_eq!(s2, &mut [1, 2, 3]);
    /// ```
    pub fn as_mut_slices_len(&mut self, start: isize, len: usize) -> (&mut [T], &mut [T]) {
        // SAFETY:
        // * All constructors ensure that the length of `vec` is greater than `0`.
        // * All constructors and other methods which modify the length ensure
        // that the length is less than or equal to `isize::MAX`.
        unsafe { inner::as_mut_slices_len(start, len, &mut self.vec) }
    }

    /// Returns two mutable slices of data in the ring buffer
    /// starting at the index `start` and with length `len`. If `len` is greater
    /// than the length of the ring buffer, then the buffer's length will be used
    /// instead, while still preserving the position of the last element.
    ///
    /// * `start` - The starting index
    /// * `len` - The length of data to read. If `len` is greater than the
    /// length of the ring buffer, then the buffer's length will be used instead, while
    /// still preserving the position of the last element.
    ///
    /// # Returns
    ///
    /// * The first slice is the starting chunk of data.
    /// * The second slice is the second contiguous chunk of data. This may
    /// or may not be empty depending if the buffer needed to wrap around to the beginning of
    /// its internal memory layout.
    ///
    /// # Performance
    ///
    /// Prefer to use this to manipulate data in bulk over indexing one element at a time.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::new(NonZeroUsize::new(4).unwrap(), 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// let (s1, s2) = rb.as_mut_slices_latest(-4, 3);
    /// assert_eq!(s1, &mut [1, 2, 3]);
    /// assert_eq!(s2, &mut []);
    ///
    /// let (s1, s2) = rb.as_mut_slices_latest(0, 5);
    /// assert_eq!(s1, &mut [2, 3, 4]);
    /// assert_eq!(s2, &mut [1]);
    /// ```
    pub fn as_mut_slices_latest(&mut self, start: isize, len: usize) -> (&mut [T], &mut [T]) {
        // SAFETY:
        // * All constructors ensure that the length of `vec` is greater than `0`.
        // * All constructors and other methods which modify the length ensure
        // that the length is less than or equal to `isize::MAX`.
        unsafe { inner::as_mut_slices_latest(start, len, &mut self.vec) }
    }

    /// Returns all the data in the buffer. The starting index will
    /// always be `0`.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::new(NonZeroUsize::new(4).unwrap(), 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// let raw_data = rb.raw_data();
    /// assert_eq!(raw_data, &[1u32, 2, 3, 4]);
    /// ```
    pub fn raw_data(&self) -> &[T] {
        &self.vec[..]
    }

    /// Returns all the data in the buffer as mutable. The starting
    /// index will always be `0`.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::new(NonZeroUsize::new(4).unwrap(), 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// let raw_data = rb.raw_data_mut();
    /// assert_eq!(raw_data, &mut [1u32, 2, 3, 4]);
    /// ```
    pub fn raw_data_mut(&mut self) -> &mut [T] {
        &mut self.vec[..]
    }

    /// Returns an immutable reference the element at the index of type `isize`.
    ///
    /// This struct is gauranteed to have at least one element.
    ///
    /// # Performance
    ///
    /// Prefer to manipulate data in bulk with methods that return slices. If you
    /// need to index multiple elements one at a time, prefer to use
    /// this over `SliceRB[i]` to reduce the number of
    /// modulo operations to perform.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::new(NonZeroUsize::new(4).unwrap(), 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// assert_eq!(*rb.get(-3), 2);
    /// ```
    #[inline]
    pub fn get(&self, i: isize) -> &T {
        // SAFETY:
        // * All constructors ensure that the length of `vec` is greater than `0`.
        // * All constructors and other methods which modify the length ensure
        // that the length is less than or equal to `isize::MAX`.
        unsafe { inner::get(i, &self.vec) }
    }

    /// Returns a mutable reference the element at the index of type `isize`.
    ///
    /// This struct is gauranteed to have at least one element.
    ///
    /// # Performance
    ///
    /// Prefer to manipulate data in bulk with methods that return slices. If you
    /// need to index multiple elements one at a time, prefer to use
    /// this over `SliceRB[i]` to reduce the number of
    /// modulo operations to perform.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::new(NonZeroUsize::new(4).unwrap(), 0);
    ///
    /// *rb.get_mut(-3) = 5;
    ///
    /// assert_eq!(*rb.get(-3), 5);
    /// ```
    #[inline]
    pub fn get_mut(&mut self, i: isize) -> &mut T {
        // SAFETY:
        // * All constructors ensure that the length of `vec` is greater than `0`.
        // * All constructors and other methods which modify the length ensure
        // that the length is less than or equal to `isize::MAX`.
        unsafe { inner::get_mut(i, &mut self.vec) }
    }

    /// Returns an immutable reference to the element at the index of type `isize`
    /// while also constraining the index `i`. This is more efficient than calling
    /// both methods individually.
    ///
    /// This struct is gauranteed to have at least one element.
    ///
    /// # Performance
    ///
    /// Prefer to manipulate data in bulk with methods that return slices. If you
    /// need to index multiple elements one at a time, prefer to use
    /// this over `SliceRB[i]` to reduce the number of
    /// modulo operations to perform.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::new(NonZeroUsize::new(4).unwrap(), 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// let mut i = -3;
    /// assert_eq!(*rb.constrain_and_get(&mut i), 2);
    /// assert_eq!(i, 1);
    /// ```
    #[inline]
    pub fn constrain_and_get(&self, i: &mut isize) -> &T {
        // SAFETY:
        // * All constructors ensure that the length of `vec` is greater than `0`.
        // * All constructors and other methods which modify the length ensure
        // that the length is less than or equal to `isize::MAX`.
        unsafe { inner::constrain_and_get(i, &self.vec) }
    }

    /// Returns a mutable reference to the element at the index of type `isize` as
    /// mutable while also constraining the index `i`. This is more efficient than
    /// calling both methods individually.
    ///
    /// This struct is gauranteed to have at least one element.
    ///
    /// # Performance
    ///
    /// Prefer to manipulate data in bulk with methods that return slices. If you
    /// need to index multiple elements one at a time, prefer to use
    /// this over `SliceRBRef[i]` to reduce the number of
    /// modulo operations to perform.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::new(NonZeroUsize::new(4).unwrap(), 0);
    ///
    /// let mut i = -3;
    /// *rb.constrain_and_get_mut(&mut i) = 2;
    ///
    /// assert_eq!(rb[1], 2);
    /// assert_eq!(i, 1);
    /// ```
    #[inline]
    pub fn constrain_and_get_mut(&mut self, i: &mut isize) -> &mut T {
        // SAFETY:
        // * All constructors ensure that the length of `vec` is greater than `0`.
        // * All constructors and other methods which modify the length ensure
        // that the length is less than or equal to `isize::MAX`.
        unsafe { inner::constrain_and_get_mut(i, &mut self.vec) }
    }
}

impl<T: Clone> SliceRB<T> {
    /// Creates a new [`SliceRB`]. All data will be initialized with the given value.
    ///
    /// * `len` - The length of the ring buffer.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let rb = SliceRB::<u32>::new(NonZeroUsize::new(3).unwrap(), 0);
    ///
    /// assert_eq!(rb.len().get(), 3);
    ///
    /// assert_eq!(rb[0], 0);
    /// assert_eq!(rb[1], 0);
    /// assert_eq!(rb[2], 0);
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if `len > isize::MAX`.
    /// * This will panic if allocation fails due to being out of memory.
    pub fn new(len: NonZeroUsize, value: T) -> Self {
        assert!(len.get() <= isize::MAX as usize);

        Self {
            vec: alloc::vec![value; len.get()],
        }
    }

    /// Creates a new [`SliceRB`] with an allocated capacity equal to exactly the
    /// given length. All data will be initialized with the given value.
    ///
    /// * `len` - The length of the ring buffer.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let rb = SliceRB::<u32>::new_exact(NonZeroUsize::new(3).unwrap(), 0);
    ///
    /// assert_eq!(rb.len().get(), 3);
    ///
    /// assert_eq!(rb[0], 0);
    /// assert_eq!(rb[1], 0);
    /// assert_eq!(rb[2], 0);
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if `len > isize::MAX`.
    /// * This will panic if allocation fails due to being out of memory.
    pub fn new_exact(len: NonZeroUsize, value: T) -> Self {
        assert!(len.get() <= isize::MAX as usize);

        let mut vec = Vec::new();
        vec.reserve_exact(len.get());
        vec.resize(len.get(), value);

        Self { vec }
    }

    /// Creates a new [`SliceRB`], while reserving extra capacity for future changes
    /// to `len`. All data from `[0..len)` will be initialized with the given value.
    ///
    /// * `len` - The length of the ring buffer.
    /// * `capacity` - The allocated capacity of the ring buffer. If this is less than
    /// `len`, then it will be ignored.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let rb = SliceRB::<u32>::with_capacity(NonZeroUsize::new(3).unwrap(), 10, 0);
    ///
    /// assert_eq!(rb.len().get(), 3);
    /// assert!(rb.capacity().get() >= 10);
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if `len > isize::MAX` or `capacity > isize::MAX`.
    /// * This will panic if allocation fails due to being out of memory.
    pub fn with_capacity(len: NonZeroUsize, capacity: usize, value: T) -> Self {
        assert!(len.get() <= isize::MAX as usize);
        assert!(capacity <= isize::MAX as usize);

        let mut vec = Vec::<T>::with_capacity(core::cmp::max(len.get(), capacity));
        vec.resize(len.get(), value);

        Self { vec }
    }

    /// Sets the length of the ring buffer while clearing all values to the given value.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::new(NonZeroUsize::new(2).unwrap(), 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    ///
    /// rb.clear_set_len(NonZeroUsize::new(4).unwrap(), 5);
    ///
    /// assert_eq!(rb.len().get(), 4);
    ///
    /// assert_eq!(rb[0], 5);
    /// assert_eq!(rb[1], 5);
    /// assert_eq!(rb[2], 5);
    /// assert_eq!(rb[3], 5);
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if `len > isize::MAX`.
    /// * This will panic if allocation fails due to being out of memory.
    pub fn clear_set_len(&mut self, len: NonZeroUsize, value: T) {
        assert!(len.get() <= isize::MAX as usize);

        self.vec.clear();
        self.vec.resize(len.get(), value);
    }

    /// Sets the length of the ring buffer.
    ///
    /// * If `len` is less than the current length, then the data
    /// will be truncated.
    /// * If `len` is larger than the current length, then all newly
    /// allocated elements appended to the end will be initialized with the
    /// given value.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::new(NonZeroUsize::new(2).unwrap(), 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    ///
    /// rb.set_len(NonZeroUsize::new(4).unwrap(), 5);
    ///
    /// assert_eq!(rb.len().get(), 4);
    ///
    /// assert_eq!(rb[0], 1);
    /// assert_eq!(rb[1], 2);
    /// assert_eq!(rb[2], 5);
    /// assert_eq!(rb[3], 5);
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if `len > isize::MAX`.
    /// * This will panic if allocation fails due to being out of memory.
    pub fn set_len(&mut self, len: NonZeroUsize, value: T) {
        assert!(len.get() <= isize::MAX as usize);

        self.vec.resize(len.get(), value);
    }
}

impl<T: Clone + Copy> SliceRB<T> {
    /// Copies the data from the ring buffer starting from the index `start`
    /// into the given slice. If the length of `slice` is larger than the
    /// length of the ring buffer, then the data will be reapeated until
    /// the given slice is filled.
    ///
    /// * `slice` - This slice to copy the data into.
    /// * `start` - The index of the ring buffer to start copying from.
    ///
    /// # Performance
    ///
    /// Prefer to use this to manipulate data in bulk over indexing one element at a time.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::new(NonZeroUsize::new(4).unwrap(), 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// let mut read_buf = [0u32; 3];
    /// rb.read_into(&mut read_buf[..], -3);
    /// assert_eq!(read_buf, [2, 3, 4]);
    ///
    /// let mut read_buf = [0u32; 9];
    /// rb.read_into(&mut read_buf[..], 2);
    /// assert_eq!(read_buf, [3, 4, 1, 2, 3, 4, 1, 2, 3]);
    /// ```
    pub fn read_into(&self, slice: &mut [T], start: isize) {
        // SAFETY:
        // * All constructors ensure that the length of `vec` is greater than `0`.
        // * All constructors and other methods which modify the length ensure
        // that the length is less than or equal to `isize::MAX`.
        unsafe { inner::read_into(slice, start, &self.vec) }
    }

    /// Copies data from the given slice into the ring buffer starting from
    /// the index `start`.
    ///
    /// Earlier data will not be copied if it will be
    /// overwritten by newer data, avoiding unecessary memcpy's. The correct
    /// placement of the newer data will still be preserved.
    ///
    /// * `slice` - This slice to copy data from.
    /// * `start` - The index of the ring buffer to start copying from.
    ///
    /// # Performance
    ///
    /// Prefer to use this to manipulate data in bulk over indexing one element at a time.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::new(NonZeroUsize::new(4).unwrap(), 0);
    ///
    /// let input = [1u32, 2, 3];
    /// rb.write_latest(&input[..], -3);
    /// assert_eq!(rb[0], 0);
    /// assert_eq!(rb[1], 1);
    /// assert_eq!(rb[2], 2);
    /// assert_eq!(rb[3], 3);
    ///
    /// let input = [1u32, 2, 3, 4, 5, 6, 7, 8, 9];
    /// rb.write_latest(&input[..], 2);
    /// assert_eq!(rb[0], 7);
    /// assert_eq!(rb[1], 8);
    /// assert_eq!(rb[2], 9);
    /// assert_eq!(rb[3], 6);
    /// ```
    pub fn write_latest(&mut self, slice: &[T], start: isize) {
        // SAFETY:
        // * All constructors ensure that the length of `vec` is greater than `0`.
        // * All constructors and other methods which modify the length ensure
        // that the length is less than or equal to `isize::MAX`.
        unsafe { inner::write_latest(slice, start, &mut self.vec) }
    }

    /// Copies data from two given slices into the ring buffer starting from
    /// the index `start`. The `first` slice will be copied first then `second`
    /// will be copied next.
    ///
    /// Earlier data will not be copied if it will be
    /// overwritten by newer data, avoiding unecessary memcpy's. The correct
    /// placement of the newer data will still be preserved.
    ///
    /// * `first` - This first slice to copy data from.
    /// * `second` - This second slice to copy data from.
    /// * `start` - The index of the ring buffer to start copying from.
    ///
    /// # Performance
    ///
    /// Prefer to use this to manipulate data in bulk over indexing one element at a time.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::num::NonZeroUsize;
    /// # use slice_ring_buf::SliceRB;
    /// let mut input_rb = SliceRB::<u32>::new(NonZeroUsize::new(4).unwrap(), 0);
    /// input_rb[0] = 1;
    /// input_rb[1] = 2;
    /// input_rb[2] = 3;
    /// input_rb[3] = 4;
    ///
    /// let mut output_rb = SliceRB::<u32>::new(NonZeroUsize::new(4).unwrap(), 0);
    /// // s1 == &[1, 2], s2 == &[]
    /// let (s1, s2) = input_rb.as_slices_len(0, 2);
    /// output_rb.write_latest_2(s1, s2, -3);
    /// assert_eq!(output_rb[0], 0);
    /// assert_eq!(output_rb[1], 1);
    /// assert_eq!(output_rb[2], 2);
    /// assert_eq!(output_rb[3], 0);
    ///
    /// let mut output_rb = SliceRB::<u32>::new(NonZeroUsize::new(2).unwrap(), 0);
    /// // s1 == &[4],  s2 == &[1, 2, 3]
    /// let (s1, s2) = input_rb.as_slices_len(3, 4);
    /// // rb[1] = 4  ->  rb[0] = 1  ->  rb[1] = 2  ->  rb[0] = 3
    /// output_rb.write_latest_2(s1, s2, 1);
    /// assert_eq!(output_rb[0], 3);
    /// assert_eq!(output_rb[1], 2);
    /// ```
    pub fn write_latest_2(&mut self, first: &[T], second: &[T], start: isize) {
        // SAFETY:
        // * All constructors ensure that the length of `vec` is greater than `0`.
        // * All constructors and other methods which modify the length ensure
        // that the length is less than or equal to `isize::MAX`.
        unsafe { inner::write_latest_2(first, second, start, &mut self.vec) }
    }
}

impl<T> core::ops::Index<isize> for SliceRB<T> {
    type Output = T;
    fn index(&self, i: isize) -> &T {
        self.get(i)
    }
}

impl<T> core::ops::IndexMut<isize> for SliceRB<T> {
    fn index_mut(&mut self, i: isize) -> &mut T {
        self.get_mut(i)
    }
}

impl<T: Clone> Clone for SliceRB<T> {
    fn clone(&self) -> Self {
        Self {
            vec: self.vec.clone(),
        }
    }
}

impl<T: Debug> Debug for SliceRB<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut f = f.debug_struct("SliceRB");
        f.field("vec", &self.vec);
        f.finish()
    }
}

impl<T> Into<Vec<T>> for SliceRB<T> {
    fn into(self) -> Vec<T> {
        self.vec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_ring_buf_initialize() {
        let ring_buf = SliceRB::<f32>::new(NonZeroUsize::new(3).unwrap(), 0.0);

        assert_eq!(&ring_buf.vec[..], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn slice_ring_buf_initialize_uninit() {
        unsafe {
            let ring_buf = SliceRB::<f32>::new_uninit(NonZeroUsize::new(3).unwrap());

            assert_eq!(ring_buf.vec.len(), 3);
        }
    }

    #[test]
    fn slice_ring_buf_clear_set_len() {
        let mut ring_buf = SliceRB::<f32>::new(NonZeroUsize::new(4).unwrap(), 0.0);
        ring_buf[0] = 1.0;
        ring_buf[1] = 2.0;
        ring_buf[2] = 3.0;
        ring_buf[3] = 4.0;

        ring_buf.clear_set_len(NonZeroUsize::new(8).unwrap(), 0.0);
        assert_eq!(ring_buf.vec.as_slice(), &[0.0; 8]);
    }

    #[test]
    fn slice_ring_buf_set_len() {
        let mut ring_buf = SliceRB::<f32>::new(NonZeroUsize::new(4).unwrap(), 0.0);
        ring_buf[0] = 1.0;
        ring_buf[1] = 2.0;
        ring_buf[2] = 3.0;
        ring_buf[3] = 4.0;

        ring_buf.set_len(NonZeroUsize::new(1).unwrap(), 0.0);
        assert_eq!(ring_buf.vec.as_slice(), &[1.0]);

        ring_buf.set_len(NonZeroUsize::new(4).unwrap(), 0.0);
        assert_eq!(ring_buf.vec.as_slice(), &[1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn slice_ring_buf_set_len_uninit() {
        let mut ring_buf = SliceRB::<f32>::new(NonZeroUsize::new(4).unwrap(), 0.0);
        ring_buf[0] = 1.0;
        ring_buf[1] = 2.0;
        ring_buf[2] = 3.0;
        ring_buf[3] = 4.0;

        unsafe {
            ring_buf.set_len_uninit(NonZeroUsize::new(1).unwrap());
        }

        assert_eq!(ring_buf.vec.as_slice(), &[1.0]);
        assert_eq!(ring_buf.vec.len(), 1);

        unsafe {
            ring_buf.set_len_uninit(NonZeroUsize::new(4).unwrap());
        }

        assert_eq!(ring_buf.vec.len(), 4);
    }

    #[test]
    fn slice_ring_buf_constrain() {
        let ring_buf = SliceRB::<f32>::new(NonZeroUsize::new(4).unwrap(), 0.0);

        assert_eq!(ring_buf.constrain(-8), 0);
        assert_eq!(ring_buf.constrain(-7), 1);
        assert_eq!(ring_buf.constrain(-6), 2);
        assert_eq!(ring_buf.constrain(-5), 3);
        assert_eq!(ring_buf.constrain(-4), 0);
        assert_eq!(ring_buf.constrain(-3), 1);
        assert_eq!(ring_buf.constrain(-2), 2);
        assert_eq!(ring_buf.constrain(-1), 3);
        assert_eq!(ring_buf.constrain(0), 0);
        assert_eq!(ring_buf.constrain(1), 1);
        assert_eq!(ring_buf.constrain(2), 2);
        assert_eq!(ring_buf.constrain(3), 3);
        assert_eq!(ring_buf.constrain(4), 0);
        assert_eq!(ring_buf.constrain(5), 1);
        assert_eq!(ring_buf.constrain(6), 2);
        assert_eq!(ring_buf.constrain(7), 3);
        assert_eq!(ring_buf.constrain(8), 0);
    }

    #[test]
    fn slice_ring_buf_index() {
        let mut ring_buf = SliceRB::<f32>::new(NonZeroUsize::new(4).unwrap(), 0.0);
        ring_buf.write_latest(&[0.0f32, 1.0, 2.0, 3.0], 0);

        let ring_buf = &ring_buf;

        assert_eq!(ring_buf[-8], 0.0);
        assert_eq!(ring_buf[-7], 1.0);
        assert_eq!(ring_buf[-6], 2.0);
        assert_eq!(ring_buf[-5], 3.0);
        assert_eq!(ring_buf[-4], 0.0);
        assert_eq!(ring_buf[-3], 1.0);
        assert_eq!(ring_buf[-2], 2.0);
        assert_eq!(ring_buf[-1], 3.0);
        assert_eq!(ring_buf[0], 0.0);
        assert_eq!(ring_buf[1], 1.0);
        assert_eq!(ring_buf[2], 2.0);
        assert_eq!(ring_buf[3], 3.0);
        assert_eq!(ring_buf[4], 0.0);
        assert_eq!(ring_buf[5], 1.0);
        assert_eq!(ring_buf[6], 2.0);
        assert_eq!(ring_buf[7], 3.0);
        assert_eq!(ring_buf[8], 0.0);
    }

    #[test]
    fn slice_ring_buf_index_mut() {
        let mut ring_buf = SliceRB::<f32>::new(NonZeroUsize::new(4).unwrap(), 0.0);
        ring_buf.write_latest(&[0.0f32, 1.0, 2.0, 3.0], 0);

        assert_eq!(&mut ring_buf[-8], &mut 0.0);
        assert_eq!(&mut ring_buf[-7], &mut 1.0);
        assert_eq!(&mut ring_buf[-6], &mut 2.0);
        assert_eq!(&mut ring_buf[-5], &mut 3.0);
        assert_eq!(&mut ring_buf[-4], &mut 0.0);
        assert_eq!(&mut ring_buf[-3], &mut 1.0);
        assert_eq!(&mut ring_buf[-2], &mut 2.0);
        assert_eq!(&mut ring_buf[-1], &mut 3.0);
        assert_eq!(&mut ring_buf[0], &mut 0.0);
        assert_eq!(&mut ring_buf[1], &mut 1.0);
        assert_eq!(&mut ring_buf[2], &mut 2.0);
        assert_eq!(&mut ring_buf[3], &mut 3.0);
        assert_eq!(&mut ring_buf[4], &mut 0.0);
        assert_eq!(&mut ring_buf[5], &mut 1.0);
        assert_eq!(&mut ring_buf[6], &mut 2.0);
        assert_eq!(&mut ring_buf[7], &mut 3.0);
        assert_eq!(&mut ring_buf[8], &mut 0.0);
    }

    #[test]
    fn slice_ring_buf_as_slices() {
        let mut ring_buf = SliceRB::<f32>::new(NonZeroUsize::new(4).unwrap(), 0.0);
        ring_buf.write_latest(&[1.0f32, 2.0, 3.0, 4.0], 0);

        let (s1, s2) = ring_buf.as_slices(0);
        assert_eq!(s1, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(s2, &[]);

        let (s1, s2) = ring_buf.as_slices(1);
        assert_eq!(s1, &[2.0, 3.0, 4.0]);
        assert_eq!(s2, &[1.0]);

        let (s1, s2) = ring_buf.as_slices(2);
        assert_eq!(s1, &[3.0, 4.0]);
        assert_eq!(s2, &[1.0, 2.0]);

        let (s1, s2) = ring_buf.as_slices(3);
        assert_eq!(s1, &[4.0]);
        assert_eq!(s2, &[1.0, 2.0, 3.0]);

        let (s1, s2) = ring_buf.as_slices(4);
        assert_eq!(s1, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(s2, &[]);
    }

    #[test]
    fn slice_ring_buf_as_mut_slices() {
        let mut ring_buf = SliceRB::<f32>::new(NonZeroUsize::new(4).unwrap(), 0.0);
        ring_buf.write_latest(&[1.0f32, 2.0, 3.0, 4.0], 0);

        let (s1, s2) = ring_buf.as_mut_slices(0);
        assert_eq!(s1, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(s2, &[]);

        let (s1, s2) = ring_buf.as_mut_slices(1);
        assert_eq!(s1, &[2.0, 3.0, 4.0]);
        assert_eq!(s2, &[1.0]);

        let (s1, s2) = ring_buf.as_mut_slices(2);
        assert_eq!(s1, &[3.0, 4.0]);
        assert_eq!(s2, &[1.0, 2.0]);

        let (s1, s2) = ring_buf.as_mut_slices(3);
        assert_eq!(s1, &[4.0]);
        assert_eq!(s2, &[1.0, 2.0, 3.0]);

        let (s1, s2) = ring_buf.as_mut_slices(4);
        assert_eq!(s1, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(s2, &[]);
    }

    #[repr(C, align(1))]
    struct Aligned1([f32; 8]);

    #[repr(C, align(2))]
    struct Aligned2([f32; 8]);

    #[repr(C, align(4))]
    struct Aligned4([f32; 8]);

    #[repr(C, align(8))]
    struct Aligned8([f32; 8]);

    #[repr(C, align(16))]
    struct Aligned16([f32; 8]);

    #[repr(C, align(32))]
    struct Aligned32([f32; 8]);

    #[repr(C, align(64))]
    struct Aligned64([f32; 8]);

    #[test]
    fn slice_ring_buf_write_latest_2() {
        let mut ring_buf = SliceRB::<f32>::new(NonZeroUsize::new(4).unwrap(), 0.0);

        ring_buf.write_latest_2(&[], &[0.0, 1.0, 2.0, 3.0, 4.0], 1);
        assert_eq!(ring_buf.vec.as_slice(), &[3.0, 4.0, 1.0, 2.0]);
        ring_buf.write_latest_2(&[-1.0], &[0.0, 1.0, 2.0, 3.0, 4.0], 1);
        assert_eq!(ring_buf.vec.as_slice(), &[2.0, 3.0, 4.0, 1.0]);
        ring_buf.write_latest_2(&[-2.0, -1.0], &[0.0, 1.0, 2.0, 3.0, 4.0], 1);
        assert_eq!(ring_buf.vec.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        ring_buf.write_latest_2(&[-2.0, -1.0], &[0.0, 1.0], 3);
        assert_eq!(ring_buf.vec.as_slice(), &[-1.0, 0.0, 1.0, -2.0]);
        ring_buf.write_latest_2(&[0.0, 1.0], &[2.0], 3);
        assert_eq!(ring_buf.vec.as_slice(), &[1.0, 2.0, 1.0, 0.0]);
        ring_buf.write_latest_2(&[1.0, 2.0, 3.0, 4.0], &[], 0);
        assert_eq!(ring_buf.vec.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        ring_buf.write_latest_2(&[1.0, 2.0], &[], 2);
        assert_eq!(ring_buf.vec.as_slice(), &[1.0, 2.0, 1.0, 2.0]);
        ring_buf.write_latest_2(&[], &[], 2);
        assert_eq!(ring_buf.vec.as_slice(), &[1.0, 2.0, 1.0, 2.0]);
        ring_buf.write_latest_2(&[1.0, 2.0, 3.0, 4.0, 5.0], &[], 1);
        assert_eq!(ring_buf.vec.as_slice(), &[4.0, 5.0, 2.0, 3.0]);
        ring_buf.write_latest_2(&[1.0, 2.0, 3.0, 4.0, 5.0], &[6.0], 2);
        assert_eq!(ring_buf.vec.as_slice(), &[3.0, 4.0, 5.0, 6.0]);
        ring_buf.write_latest_2(&[1.0, 2.0, 3.0, 4.0, 5.0], &[6.0, 7.0], 2);
        assert_eq!(ring_buf.vec.as_slice(), &[7.0, 4.0, 5.0, 6.0]);
        ring_buf.write_latest_2(&[1.0, 2.0, 3.0, 4.0, 5.0], &[6.0, 7.0, 8.0, 9.0, 10.0], 3);
        assert_eq!(ring_buf.vec.as_slice(), &[10.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn slice_ring_buf_write_latest() {
        let mut ring_buf = SliceRB::<f32>::new(NonZeroUsize::new(4).unwrap(), 0.0);

        let input = [0.0f32, 1.0, 2.0, 3.0];

        ring_buf.write_latest(&input, 0);
        assert_eq!(ring_buf.vec.as_slice(), &[0.0, 1.0, 2.0, 3.0]);
        ring_buf.write_latest(&input, 1);
        assert_eq!(ring_buf.vec.as_slice(), &[3.0, 0.0, 1.0, 2.0]);
        ring_buf.write_latest(&input, 2);
        assert_eq!(ring_buf.vec.as_slice(), &[2.0, 3.0, 0.0, 1.0]);
        ring_buf.write_latest(&input, 3);
        assert_eq!(ring_buf.vec.as_slice(), &[1.0, 2.0, 3.0, 0.0]);
        ring_buf.write_latest(&input, 4);
        assert_eq!(ring_buf.vec.as_slice(), &[0.0, 1.0, 2.0, 3.0]);

        let input = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        ring_buf.write_latest(&input, 0);
        assert_eq!(ring_buf.vec.as_slice(), &[4.0, 5.0, 6.0, 7.0]);
        ring_buf.write_latest(&input, 1);
        assert_eq!(ring_buf.vec.as_slice(), &[7.0, 4.0, 5.0, 6.0]);
        ring_buf.write_latest(&input, 2);
        assert_eq!(ring_buf.vec.as_slice(), &[6.0, 7.0, 4.0, 5.0]);
        ring_buf.write_latest(&input, 3);
        assert_eq!(ring_buf.vec.as_slice(), &[5.0, 6.0, 7.0, 4.0]);
        ring_buf.write_latest(&input, 4);
        assert_eq!(ring_buf.vec.as_slice(), &[4.0, 5.0, 6.0, 7.0]);

        let input = [0.0f32, 1.0];

        ring_buf.write_latest(&input, 0);
        assert_eq!(ring_buf.vec.as_slice(), &[0.0, 1.0, 6.0, 7.0]);
        ring_buf.write_latest(&input, 1);
        assert_eq!(ring_buf.vec.as_slice(), &[0.0, 0.0, 1.0, 7.0]);
        ring_buf.write_latest(&input, 2);
        assert_eq!(ring_buf.vec.as_slice(), &[0.0, 0.0, 0.0, 1.0]);
        ring_buf.write_latest(&input, 3);
        assert_eq!(ring_buf.vec.as_slice(), &[1.0, 0.0, 0.0, 0.0]);
        ring_buf.write_latest(&input, 4);
        assert_eq!(ring_buf.vec.as_slice(), &[0.0, 1.0, 0.0, 0.0]);

        let aligned_input = Aligned1([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.vec.as_slice(), &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned2([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.vec.as_slice(), &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned4([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.vec.as_slice(), &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned8([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.vec.as_slice(), &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned16([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.vec.as_slice(), &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned32([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.vec.as_slice(), &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned64([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.vec.as_slice(), &[12.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn slice_ring_buf_as_slices_len() {
        let mut ring_buf = SliceRB::<f32>::new(NonZeroUsize::new(4).unwrap(), 0.0);
        ring_buf.write_latest(&[0.0, 1.0, 2.0, 3.0], 0);

        let (s1, s2) = ring_buf.as_slices_len(0, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(0, 1);
        assert_eq!(s1, &[0.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(0, 2);
        assert_eq!(s1, &[0.0, 1.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(0, 3);
        assert_eq!(s1, &[0.0, 1.0, 2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(0, 4);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(0, 5);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);

        let (s1, s2) = ring_buf.as_slices_len(1, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(1, 1);
        assert_eq!(s1, &[1.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(1, 2);
        assert_eq!(s1, &[1.0, 2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(1, 3);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(1, 4);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_slices_len(1, 5);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[0.0]);

        let (s1, s2) = ring_buf.as_slices_len(2, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(2, 1);
        assert_eq!(s1, &[2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(2, 2);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(2, 3);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_slices_len(2, 4);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0, 1.0]);
        let (s1, s2) = ring_buf.as_slices_len(2, 5);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0, 1.0]);

        let (s1, s2) = ring_buf.as_slices_len(3, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(3, 1);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(3, 2);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_slices_len(3, 3);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0]);
        let (s1, s2) = ring_buf.as_slices_len(3, 4);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_slices_len(3, 5);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0, 2.0]);

        let (s1, s2) = ring_buf.as_slices_len(4, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(4, 1);
        assert_eq!(s1, &[0.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(4, 2);
        assert_eq!(s1, &[0.0, 1.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(4, 3);
        assert_eq!(s1, &[0.0, 1.0, 2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(4, 4);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(4, 5);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
    }

    #[test]
    fn slice_ring_buf_as_slices_latest() {
        let mut ring_buf = SliceRB::<f32>::new(NonZeroUsize::new(4).unwrap(), 0.0);
        ring_buf.write_latest(&[0.0, 1.0, 2.0, 3.0], 0);

        let (s1, s2) = ring_buf.as_slices_latest(0, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(0, 1);
        assert_eq!(s1, &[0.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(0, 2);
        assert_eq!(s1, &[0.0, 1.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(0, 3);
        assert_eq!(s1, &[0.0, 1.0, 2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(0, 4);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(0, 5);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_slices_latest(0, 6);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0, 1.0]);
        let (s1, s2) = ring_buf.as_slices_latest(0, 7);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_slices_latest(0, 10);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0, 1.0]);

        let (s1, s2) = ring_buf.as_slices_latest(1, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(1, 1);
        assert_eq!(s1, &[1.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(1, 2);
        assert_eq!(s1, &[1.0, 2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(1, 3);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(1, 4);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_slices_latest(1, 5);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0, 1.0]);
        let (s1, s2) = ring_buf.as_slices_latest(1, 6);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_slices_latest(1, 7);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(1, 10);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0, 2.0]);

        let (s1, s2) = ring_buf.as_slices_latest(2, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(2, 1);
        assert_eq!(s1, &[2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(2, 2);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(2, 3);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_slices_latest(2, 4);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0, 1.0]);
        let (s1, s2) = ring_buf.as_slices_latest(2, 5);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_slices_latest(2, 6);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(2, 7);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_slices_latest(2, 10);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);

        let (s1, s2) = ring_buf.as_slices_latest(3, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(3, 1);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(3, 2);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_slices_latest(3, 3);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0]);
        let (s1, s2) = ring_buf.as_slices_latest(3, 4);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_slices_latest(3, 5);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(3, 6);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_slices_latest(3, 7);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0, 1.0]);
        let (s1, s2) = ring_buf.as_slices_latest(3, 10);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[0.0]);

        let (s1, s2) = ring_buf.as_slices_latest(4, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(4, 1);
        assert_eq!(s1, &[0.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(4, 2);
        assert_eq!(s1, &[0.0, 1.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(4, 3);
        assert_eq!(s1, &[0.0, 1.0, 2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(4, 4);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(4, 5);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_slices_latest(4, 6);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0, 1.0]);
        let (s1, s2) = ring_buf.as_slices_latest(4, 7);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_slices_latest(4, 10);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0, 1.0]);
    }

    #[test]
    fn slice_ring_buf_as_mut_slices_len() {
        let mut ring_buf = SliceRB::<f32>::new(NonZeroUsize::new(4).unwrap(), 0.0);
        ring_buf.write_latest(&[0.0, 1.0, 2.0, 3.0], 0);

        let (s1, s2) = ring_buf.as_mut_slices_len(0, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(0, 1);
        assert_eq!(s1, &[0.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(0, 2);
        assert_eq!(s1, &[0.0, 1.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(0, 3);
        assert_eq!(s1, &[0.0, 1.0, 2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(0, 4);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(0, 5);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);

        let (s1, s2) = ring_buf.as_mut_slices_len(1, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(1, 1);
        assert_eq!(s1, &[1.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(1, 2);
        assert_eq!(s1, &[1.0, 2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(1, 3);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(1, 4);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_mut_slices_len(1, 5);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[0.0]);

        let (s1, s2) = ring_buf.as_mut_slices_len(2, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(2, 1);
        assert_eq!(s1, &[2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(2, 2);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(2, 3);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_mut_slices_len(2, 4);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0, 1.0]);
        let (s1, s2) = ring_buf.as_mut_slices_len(2, 5);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0, 1.0]);

        let (s1, s2) = ring_buf.as_mut_slices_len(3, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(3, 1);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(3, 2);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_mut_slices_len(3, 3);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0]);
        let (s1, s2) = ring_buf.as_mut_slices_len(3, 4);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_mut_slices_len(3, 5);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0, 2.0]);

        let (s1, s2) = ring_buf.as_mut_slices_len(4, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(4, 1);
        assert_eq!(s1, &[0.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(4, 2);
        assert_eq!(s1, &[0.0, 1.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(4, 3);
        assert_eq!(s1, &[0.0, 1.0, 2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(4, 4);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(4, 5);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
    }

    #[test]
    fn slice_ring_buf_as_mut_slices_latest() {
        let mut ring_buf = SliceRB::<f32>::new(NonZeroUsize::new(4).unwrap(), 0.0);
        ring_buf.write_latest(&[0.0, 1.0, 2.0, 3.0], 0);

        let (s1, s2) = ring_buf.as_mut_slices_latest(0, 0);
        assert_eq!(s1, &mut []);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(0, 1);
        assert_eq!(s1, &mut [0.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(0, 2);
        assert_eq!(s1, &mut [0.0, 1.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(0, 3);
        assert_eq!(s1, &mut [0.0, 1.0, 2.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(0, 4);
        assert_eq!(s1, &mut [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(0, 5);
        assert_eq!(s1, &mut [1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut [0.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(0, 6);
        assert_eq!(s1, &mut [2.0, 3.0]);
        assert_eq!(s2, &mut [0.0, 1.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(0, 7);
        assert_eq!(s1, &mut [3.0]);
        assert_eq!(s2, &mut [0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(0, 10);
        assert_eq!(s1, &mut [2.0, 3.0]);
        assert_eq!(s2, &mut [0.0, 1.0]);

        let (s1, s2) = ring_buf.as_mut_slices_latest(1, 0);
        assert_eq!(s1, &mut []);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(1, 1);
        assert_eq!(s1, &mut [1.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(1, 2);
        assert_eq!(s1, &mut [1.0, 2.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(1, 3);
        assert_eq!(s1, &mut [1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(1, 4);
        assert_eq!(s1, &mut [1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut [0.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(1, 5);
        assert_eq!(s1, &mut [2.0, 3.0]);
        assert_eq!(s2, &mut [0.0, 1.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(1, 6);
        assert_eq!(s1, &mut [3.0]);
        assert_eq!(s2, &mut [0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(1, 7);
        assert_eq!(s1, &mut [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(1, 10);
        assert_eq!(s1, &mut [3.0]);
        assert_eq!(s2, &mut [0.0, 1.0, 2.0]);

        let (s1, s2) = ring_buf.as_mut_slices_latest(2, 0);
        assert_eq!(s1, &mut []);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(2, 1);
        assert_eq!(s1, &mut [2.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(2, 2);
        assert_eq!(s1, &mut [2.0, 3.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(2, 3);
        assert_eq!(s1, &mut [2.0, 3.0]);
        assert_eq!(s2, &mut [0.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(2, 4);
        assert_eq!(s1, &mut [2.0, 3.0]);
        assert_eq!(s2, &mut [0.0, 1.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(2, 5);
        assert_eq!(s1, &mut [3.0]);
        assert_eq!(s2, &mut [0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(2, 6);
        assert_eq!(s1, &mut [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(2, 7);
        assert_eq!(s1, &mut [1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut [0.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(2, 10);
        assert_eq!(s1, &mut [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut []);

        let (s1, s2) = ring_buf.as_mut_slices_latest(3, 0);
        assert_eq!(s1, &mut []);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(3, 1);
        assert_eq!(s1, &mut [3.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(3, 2);
        assert_eq!(s1, &mut [3.0]);
        assert_eq!(s2, &mut [0.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(3, 3);
        assert_eq!(s1, &mut [3.0]);
        assert_eq!(s2, &mut [0.0, 1.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(3, 4);
        assert_eq!(s1, &mut [3.0]);
        assert_eq!(s2, &mut [0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(3, 5);
        assert_eq!(s1, &mut [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(3, 6);
        assert_eq!(s1, &mut [1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut [0.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(3, 7);
        assert_eq!(s1, &mut [2.0, 3.0]);
        assert_eq!(s2, &mut [0.0, 1.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(3, 10);
        assert_eq!(s1, &mut [1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut [0.0]);

        let (s1, s2) = ring_buf.as_mut_slices_latest(4, 0);
        assert_eq!(s1, &mut []);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(4, 1);
        assert_eq!(s1, &mut [0.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(4, 2);
        assert_eq!(s1, &mut [0.0, 1.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(4, 3);
        assert_eq!(s1, &mut [0.0, 1.0, 2.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(4, 4);
        assert_eq!(s1, &mut [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(4, 5);
        assert_eq!(s1, &mut [1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut [0.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(4, 6);
        assert_eq!(s1, &mut [2.0, 3.0]);
        assert_eq!(s2, &mut [0.0, 1.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(4, 7);
        assert_eq!(s1, &mut [3.0]);
        assert_eq!(s2, &mut [0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(4, 10);
        assert_eq!(s1, &mut [2.0, 3.0]);
        assert_eq!(s2, &mut [0.0, 1.0]);
    }

    #[test]
    fn slice_ring_buf_read_into() {
        let mut ring_buf = SliceRB::<f32>::new(NonZeroUsize::new(4).unwrap(), 0.0);
        ring_buf.write_latest(&[0.0, 1.0, 2.0, 3.0], 0);

        let mut output = [0.0f32; 4];

        ring_buf.read_into(&mut output, 0);
        assert_eq!(output, [0.0, 1.0, 2.0, 3.0]);
        ring_buf.read_into(&mut output, 1);
        assert_eq!(output, [1.0, 2.0, 3.0, 0.0]);
        ring_buf.read_into(&mut output, 2);
        assert_eq!(output, [2.0, 3.0, 0.0, 1.0]);
        ring_buf.read_into(&mut output, 3);
        assert_eq!(output, [3.0, 0.0, 1.0, 2.0]);
        ring_buf.read_into(&mut output, 4);
        assert_eq!(output, [0.0, 1.0, 2.0, 3.0]);

        let mut output = [0.0f32; 3];

        ring_buf.read_into(&mut output, 0);
        assert_eq!(output, [0.0, 1.0, 2.0]);
        ring_buf.read_into(&mut output, 1);
        assert_eq!(output, [1.0, 2.0, 3.0]);
        ring_buf.read_into(&mut output, 2);
        assert_eq!(output, [2.0, 3.0, 0.0]);
        ring_buf.read_into(&mut output, 3);
        assert_eq!(output, [3.0, 0.0, 1.0]);
        ring_buf.read_into(&mut output, 4);
        assert_eq!(output, [0.0, 1.0, 2.0]);

        let mut output = [0.0f32; 5];

        ring_buf.read_into(&mut output, 0);
        assert_eq!(output, [0.0, 1.0, 2.0, 3.0, 0.0]);
        ring_buf.read_into(&mut output, 1);
        assert_eq!(output, [1.0, 2.0, 3.0, 0.0, 1.0]);
        ring_buf.read_into(&mut output, 2);
        assert_eq!(output, [2.0, 3.0, 0.0, 1.0, 2.0]);
        ring_buf.read_into(&mut output, 3);
        assert_eq!(output, [3.0, 0.0, 1.0, 2.0, 3.0]);
        ring_buf.read_into(&mut output, 4);
        assert_eq!(output, [0.0, 1.0, 2.0, 3.0, 0.0]);

        let mut output = [0.0f32; 10];

        ring_buf.read_into(&mut output, 0);
        assert_eq!(output, [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0]);
        ring_buf.read_into(&mut output, 3);
        assert_eq!(output, [3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0]);

        let mut aligned_output = Aligned1([0.0; 8]);
        ring_buf.read_into(&mut aligned_output.0, 0);
        assert_eq!(aligned_output.0, [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);

        let mut aligned_output = Aligned2([0.0; 8]);
        ring_buf.read_into(&mut aligned_output.0, 0);
        assert_eq!(aligned_output.0, [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);

        let mut aligned_output = Aligned4([0.0; 8]);
        ring_buf.read_into(&mut aligned_output.0, 0);
        assert_eq!(aligned_output.0, [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);

        let mut aligned_output = Aligned8([0.0; 8]);
        ring_buf.read_into(&mut aligned_output.0, 0);
        assert_eq!(aligned_output.0, [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);

        let mut aligned_output = Aligned16([0.0; 8]);
        ring_buf.read_into(&mut aligned_output.0, 0);
        assert_eq!(aligned_output.0, [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);

        let mut aligned_output = Aligned32([0.0; 8]);
        ring_buf.read_into(&mut aligned_output.0, 0);
        assert_eq!(aligned_output.0, [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);

        let mut aligned_output = Aligned64([0.0; 8]);
        ring_buf.read_into(&mut aligned_output.0, 0);
        assert_eq!(aligned_output.0, [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);
    }
}
