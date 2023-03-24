//! A ring buffer implementation that is optimized for working with slices. Note this
//! pretty much does the same thing as [`VecDeque`], but with the added ability to
//! index using negative values, as well as working with buffers allocated on the
//! stack.
//!
//! This crate has no consumer/producer logic, and is meant to be used as a raw data
//! structure or a base for other data structures.
//!
//! This is optimized for manipulating data in chunks with slices. If your algorithm
//! instead indexes elements one at a time and only uses buffers that have a size
//! that is a power of two, then consider my crate [`bit_mask_ring_buf`].
//!
//! ## Installation
//! Add `slice_ring_buf` as a dependency in your `Cargo.toml`:
//! ```toml
//! slice_ring_buf = 0.2
//! ```
//!
//! ## Example
//! ```rust
//! use slice_ring_buf::{SliceRB, SliceRbRef};
//!
//! // Create a ring buffer with type u32. The data will be
//! // initialized with the default value (0 in this case).
//! let mut rb = SliceRB::<u32>::from_len(4);
//!
//! // Memcpy data from a slice into the ring buffer at arbitrary
//! // `isize` indexes. Earlier data will not be copied if it will
//! // be overwritten by newer data, avoiding unecessary memcpy's.
//! // The correct placement of the newer data will still be preserved.
//! rb.write_latest(&[0, 2, 3, 4, 1], 0);
//! assert_eq!(rb[0], 1);
//! assert_eq!(rb[1], 2);
//! assert_eq!(rb[2], 3);
//! assert_eq!(rb[3], 4);
//!
//! // Memcpy into slices at arbitrary `isize` indexes and length.
//! let mut read_buffer = [0u32; 7];
//! rb.read_into(&mut read_buffer, 2);
//! assert_eq!(read_buffer, [3, 4, 1, 2, 3, 4, 1]);
//!
//! // Read/write by retrieving slices directly.
//! let (s1, s2) = rb.as_slices_len(1, 4);
//! assert_eq!(s1, &[2, 3, 4]);
//! assert_eq!(s2, &[1]);
//!
//! // Read/write to buffer by indexing. Performance will be limited
//! // by the modulo (remainder) operation on an isize value.
//! rb[0] = 0;
//! rb[1] = 1;
//! rb[2] = 2;
//! rb[3] = 3;
//!
//! // Wrap when reading/writing outside of bounds. Performance will be
//! // limited by the modulo (remainder) operation on an isize value.
//! assert_eq!(rb[-1], 3);
//! assert_eq!(rb[10], 2);
//!
//! // Aligned/stack data may also be used.
//! let mut stack_data = [0u32, 1, 2, 3];
//! let mut rb_ref = SliceRbRef::new(&mut stack_data);
//! rb_ref[-4] = 5;
//! let (s1, s2) = rb_ref.as_slices_len(0, 3);
//! assert_eq!(s1, &[5, 1, 2]);
//! assert_eq!(s2, &[]);
//! ```
//!
//! [`VecDeque`]: https://doc.rust-lang.org/std/collections/struct.VecDeque.html
//! [`bit_mask_ring_buf`]: https://crates.io/crates/bit_mask_ring_buf

mod referenced;
pub use referenced::SliceRbRef;

/// A fast ring buffer implementation optimized for working with slices.
/// Copies/reads with slices are implemented with memcpy.
#[derive(Debug, Clone)]
pub struct SliceRB<T: Copy + Clone + Default> {
    vec: Vec<T>,
    len_isize: isize,
}

impl<T: Copy + Clone + Default> SliceRB<T> {
    /// Creates a new [`SliceRB`]. All data will be initialized with the default value.
    ///
    /// * `len` - The length of the ring buffer.
    ///
    /// # Example
    ///
    /// ```
    /// use slice_ring_buf::SliceRB;
    ///
    /// let rb = SliceRB::<u32>::from_len(3);
    ///
    /// assert_eq!(rb.len(), 3);
    ///
    /// assert_eq!(rb[0], 0);
    /// assert_eq!(rb[1], 0);
    /// assert_eq!(rb[2], 0);
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if `len = 0`.
    /// * This will panic if this tries to allocate more than `isize::MAX` bytes.
    ///
    /// [`SliceRB`]: struct.SliceRB.html
    pub fn from_len(len: usize) -> Self {
        assert_ne!(len, 0);

        Self {
            vec: vec![Default::default(); len],
            len_isize: len as isize,
        }
    }

    /// Creates a new [`SliceRB`], while reserving extra capacity for future changes
    /// to `len`. All data from `[0..len)` will be initialized with the default value.
    ///
    /// * `len` - The length of the ring buffer.
    /// * `capacity` - The allocated capacity of the ring buffer. If this is less than
    /// `len`, then it will be ignored.
    ///
    /// # Example
    ///
    /// ```
    /// use slice_ring_buf::SliceRB;
    ///
    /// let rb = SliceRB::<u32>::from_len_with_capacity(3, 10);
    ///
    /// assert_eq!(rb.len(), 3);
    /// assert!(rb.capacity() >= 10);
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if `len = 0`.
    /// * This will panic if this tries to allocate more than `isize::MAX` bytes.
    ///
    /// [`SliceRB`]: struct.SliceRB.html
    pub fn from_len_with_capacity(len: usize, capacity: usize) -> Self {
        assert_ne!(len, 0);

        let mut vec = Vec::<T>::with_capacity(std::cmp::max(len, capacity));
        vec.resize(len, Default::default());

        Self {
            vec,
            len_isize: len as isize,
        }
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
    /// use slice_ring_buf::SliceRB;
    ///
    /// unsafe {
    ///     let rb = SliceRB::<u32>::from_len_uninit(3);
    ///     assert_eq!(rb.len(), 3);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if `len = 0`.
    /// * This will panic if this tries to allocate more than `isize::MAX` bytes.
    ///
    /// [`SliceRB`]: struct.SliceRB.html
    pub unsafe fn from_len_uninit(len: usize) -> Self {
        assert_ne!(len, 0);

        let mut vec = Vec::with_capacity(len);
        vec.set_len(len);

        Self {
            vec,
            len_isize: len as isize,
        }
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
    /// use slice_ring_buf::SliceRB;
    ///
    /// unsafe {
    ///     let rb = SliceRB::<u32>::from_len_with_capacity_uninit(3, 10);
    ///     assert_eq!(rb.len(), 3);
    ///     assert!(rb.capacity() >= 10);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if `len = 0`.
    /// * This will panic if this tries to allocate more than `isize::MAX` bytes.
    ///
    /// [`SliceRB`]: struct.SliceRB.html
    pub unsafe fn from_len_with_capacity_uninit(len: usize, capacity: usize) -> Self {
        assert_ne!(len, 0);

        let mut vec = Vec::with_capacity(std::cmp::max(len, capacity));
        vec.set_len(len);
        Self {
            vec,
            len_isize: len as isize,
        }
    }

    /// Sets the length of the ring buffer while clearing all values to the default value.
    ///
    /// # Example
    ///
    /// ```
    /// use slice_ring_buf::SliceRB;
    ///
    /// let mut rb = SliceRB::<u32>::from_len(2);
    /// rb[0] = 1;
    /// rb[1] = 2;
    ///
    /// rb.clear_set_len(4);
    ///
    /// assert_eq!(rb.len(), 4);
    ///
    /// assert_eq!(rb[0], 0);
    /// assert_eq!(rb[1], 0);
    /// assert_eq!(rb[2], 0);
    /// assert_eq!(rb[3], 0);
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if `len = 0`.
    /// * This will panic if this tries to allocate more than `isize::MAX` bytes.
    pub fn clear_set_len(&mut self, len: usize) {
        assert_ne!(len, 0);

        self.vec.clear();
        self.vec.resize(len, Default::default());
        self.len_isize = len as isize;
    }

    /// Sets the length of the ring buffer.
    ///
    /// * If `len` is less than the current length, then the data
    /// will be truncated.
    /// * If `len` is larger than the current length, then all newly
    /// allocated elements appended to the end will be initialized with the
    /// default value.
    ///
    /// # Example
    ///
    /// ```
    /// use slice_ring_buf::SliceRB;
    ///
    /// let mut rb = SliceRB::<u32>::from_len(2);
    /// rb[0] = 1;
    /// rb[1] = 2;
    ///
    /// rb.set_len(4);
    ///
    /// assert_eq!(rb.len(), 4);
    ///
    /// assert_eq!(rb[0], 1);
    /// assert_eq!(rb[1], 2);
    /// assert_eq!(rb[2], 0);
    /// assert_eq!(rb[3], 0);
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if `len = 0`.
    /// * This will panic if this tries to allocate more than `isize::MAX` bytes.
    pub fn set_len(&mut self, len: usize) {
        assert_ne!(len, 0);

        self.vec.resize(len, Default::default());
        self.len_isize = len as isize;
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
    /// use slice_ring_buf::SliceRB;
    ///
    /// let mut rb = SliceRB::<u32>::from_len(2);
    /// rb[0] = 1;
    /// rb[1] = 2;
    ///
    /// unsafe {
    ///     rb.set_len_uninit(4);
    ///
    ///     assert_eq!(rb.len(), 4);
    ///
    ///     assert_eq!(rb[0], 1);
    ///     assert_eq!(rb[1], 2);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if `len = 0`.
    /// * This will panic if this tries to allocate more than `isize::MAX` bytes.
    pub unsafe fn set_len_uninit(&mut self, len: usize) {
        assert_ne!(len, 0);

        if len != self.vec.len() {
            if len < self.vec.len() {
                // Truncate data.
                self.vec.resize(len, Default::default());
            } else {
                // Extend without initializing.
                self.vec.reserve_exact(len - self.vec.len());
                self.vec.set_len(len);
            }
            self.len_isize = len as isize;
        }
    }

    /// Clears all values in the ring buffer to the default value.
    ///
    /// # Example
    ///
    /// ```
    /// use slice_ring_buf::SliceRB;
    ///
    /// let mut rb = SliceRB::<u32>::from_len(2);
    /// rb[0] = 1;
    /// rb[1] = 2;
    ///
    /// rb.clear();
    ///
    /// assert_eq!(rb[0], 0);
    /// assert_eq!(rb[1], 0);
    /// ```
    pub fn clear(&mut self) {
        let len = self.vec.len();
        self.vec.clear();
        self.vec.resize(len, Default::default());
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
    /// use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::from_len(2);
    ///
    /// rb.reserve(8);
    ///
    /// assert!(rb.capacity() >= 10);
    /// ```
    ///
    /// # Panics
    ///
    /// * Panics if the new capacity overflows `usize`.
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
    /// use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::from_len(2);
    ///
    /// rb.reserve_exact(8);
    ///
    /// assert!(rb.capacity() >= 10);
    /// ```
    ///
    /// # Panics
    ///
    /// * Panics if the new capacity overflows `usize`.
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
    /// use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::from_len(2);
    ///
    /// rb.reserve(8);
    /// assert!(rb.capacity() >= 10);
    ///
    /// rb.shrink_to_fit();
    /// assert!(rb.capacity() >= 2);
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
    /// use slice_ring_buf::SliceRB;
    ///
    /// let mut rb = SliceRB::<u32>::from_len(4);
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
        let start = self.constrain(start) as usize;

        // Safe because self.constrain() is always in range.
        //
        // Both the length of self.vec and the value of self.mask are only modified
        // in self.set_len(). This function makes sure these values are valid.
        // The constructors also correctly call this function.
        //
        // Memory is created and initialized by a Vec, so it is always valid.
        unsafe {
            let self_vec_ptr = self.vec.as_ptr();
            (
                &*std::ptr::slice_from_raw_parts(self_vec_ptr.add(start), self.vec.len() - start),
                &*std::ptr::slice_from_raw_parts(self_vec_ptr, start),
            )
        }
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
    /// use slice_ring_buf::SliceRB;
    ///
    /// let mut rb = SliceRB::<u32>::from_len(4);
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
        let start = self.constrain(start) as usize;

        // Safe because self.constrain() is always in range.
        //
        // Both the length of self.vec and the value of self.mask are only modified
        // in self.set_len(). This function makes sure these values are valid.
        // The constructors also correctly call this function.
        //
        // Memory is created and initialized by a Vec, so it is always valid.
        unsafe {
            let self_vec_ptr = self.vec.as_ptr();

            let first_portion_len = self.vec.len() - start;
            if len > first_portion_len {
                let second_portion_len = std::cmp::min(len - first_portion_len, start);
                (
                    &*std::ptr::slice_from_raw_parts(self_vec_ptr.add(start), first_portion_len),
                    &*std::ptr::slice_from_raw_parts(self_vec_ptr, second_portion_len),
                )
            } else {
                (
                    &*std::ptr::slice_from_raw_parts(self_vec_ptr.add(start), len),
                    &[],
                )
            }
        }
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
    /// use slice_ring_buf::SliceRB;
    ///
    /// let mut rb = SliceRB::<u32>::from_len(4);
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
        // Safe because of the algorithm of bit-masking the index on an array/vec
        // whose length is a power of 2.
        //
        // Both the length of self.vec and the value of self.mask are only modified
        // in self.set_len(). This function makes sure these values are valid.
        // The constructors also correctly call this function.
        //
        // Memory is created and initialized by a Vec, so it is always valid.
        unsafe {
            let self_vec_ptr = self.vec.as_ptr();

            if len > self.vec.len() {
                let end_index = start + len as isize;
                let start = self.constrain(end_index - self.vec.len() as isize) as usize;

                (
                    &*std::ptr::slice_from_raw_parts(
                        self_vec_ptr.add(start),
                        self.vec.len() - start,
                    ),
                    &*std::ptr::slice_from_raw_parts(self_vec_ptr, start),
                )
            } else {
                let start = self.constrain(start) as usize;
                let first_portion_len = self.vec.len() - start;
                if len > first_portion_len {
                    let second_portion_len = std::cmp::min(len - first_portion_len, start);
                    (
                        &*std::ptr::slice_from_raw_parts(
                            self_vec_ptr.add(start),
                            first_portion_len,
                        ),
                        &*std::ptr::slice_from_raw_parts(self_vec_ptr, second_portion_len),
                    )
                } else {
                    (
                        &*std::ptr::slice_from_raw_parts(self_vec_ptr.add(start), len),
                        &[],
                    )
                }
            }
        }
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
    /// use slice_ring_buf::SliceRB;
    ///
    /// let mut rb = SliceRB::<u32>::from_len(4);
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
        let start = self.constrain(start) as usize;

        // Safe because self.constrain() is always in range.
        //
        // Both the length of self.vec and the value of self.mask are only modified
        // in self.set_len(). This function makes sure these values are valid.
        // The constructors also correctly call this function.
        //
        // Memory is created and initialized by a Vec, so it is always valid.
        unsafe {
            let self_vec_ptr = self.vec.as_mut_ptr();
            (
                &mut *std::ptr::slice_from_raw_parts_mut(
                    self_vec_ptr.add(start),
                    self.vec.len() - start,
                ),
                &mut *std::ptr::slice_from_raw_parts_mut(self_vec_ptr, start),
            )
        }
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
    /// use slice_ring_buf::SliceRB;
    ///
    /// let mut rb = SliceRB::<u32>::from_len(4);
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
        let start = self.constrain(start) as usize;

        // Safe because self.constrain() is always in range.
        //
        // Both the length of self.vec and the value of self.mask are only modified
        // in self.set_len(). This function makes sure these values are valid.
        // The constructors also correctly call this function.
        //
        // Memory is created and initialized by a Vec, so it is always valid.
        unsafe {
            let self_vec_ptr = self.vec.as_mut_ptr();

            let first_portion_len = self.vec.len() - start;
            if len > first_portion_len {
                let second_portion_len = std::cmp::min(len - first_portion_len, start);
                (
                    &mut *std::ptr::slice_from_raw_parts_mut(
                        self_vec_ptr.add(start),
                        first_portion_len,
                    ),
                    &mut *std::ptr::slice_from_raw_parts_mut(self_vec_ptr, second_portion_len),
                )
            } else {
                (
                    &mut *std::ptr::slice_from_raw_parts_mut(self_vec_ptr.add(start), len),
                    &mut [],
                )
            }
        }
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
    /// use slice_ring_buf::SliceRB;
    ///
    /// let mut rb = SliceRB::<u32>::from_len(4);
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
        // Safe because of the algorithm of bit-masking the index on an array/vec
        // whose length is a power of 2.
        //
        // Both the length of self.vec and the value of self.mask are only modified
        // in self.set_len(). This function makes sure these values are valid.
        // The constructors also correctly call this function.
        //
        // Memory is created and initialized by a Vec, so it is always valid.
        unsafe {
            let self_vec_ptr = self.vec.as_mut_ptr();

            if len > self.vec.len() {
                let end_index = start + len as isize;
                let start = self.constrain(end_index - self.vec.len() as isize) as usize;

                (
                    &mut *std::ptr::slice_from_raw_parts_mut(
                        self_vec_ptr.add(start),
                        self.vec.len() - start,
                    ),
                    &mut *std::ptr::slice_from_raw_parts_mut(self_vec_ptr, start),
                )
            } else {
                let start = self.constrain(start) as usize;
                let first_portion_len = self.vec.len() - start;
                if len > first_portion_len {
                    let second_portion_len = std::cmp::min(len - first_portion_len, start);
                    (
                        &mut *std::ptr::slice_from_raw_parts_mut(
                            self_vec_ptr.add(start),
                            first_portion_len,
                        ),
                        &mut *std::ptr::slice_from_raw_parts_mut(self_vec_ptr, second_portion_len),
                    )
                } else {
                    (
                        &mut *std::ptr::slice_from_raw_parts_mut(self_vec_ptr.add(start), len),
                        &mut [],
                    )
                }
            }
        }
    }

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
    /// use slice_ring_buf::SliceRB;
    ///
    /// let mut rb = SliceRB::<u32>::from_len(4);
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
        let start = self.constrain(start) as usize;

        // Safe because self.constrain() is always in range.
        //
        // Memory is created and initialized by a Vec, so it is always valid.
        //
        // Memory cannot overlap because a mutable and immutable reference do not
        // alias.
        unsafe {
            let self_vec_ptr = self.vec.as_ptr();
            let mut slice_ptr = slice.as_mut_ptr();
            let mut slice_len = slice.len();

            // While slice is longer than from start to the end of self.vec,
            // copy that first portion, then wrap to the beginning and copy the
            // second portion up to start.
            let first_portion_len = self.vec.len() - start;
            while slice_len > first_portion_len {
                // Copy first portion
                std::ptr::copy_nonoverlapping(
                    self_vec_ptr.add(start),
                    slice_ptr,
                    first_portion_len,
                );
                slice_ptr = slice_ptr.add(first_portion_len);
                slice_len -= first_portion_len;

                // Copy second portion
                let second_portion_len = std::cmp::min(slice_len, start);
                std::ptr::copy_nonoverlapping(self_vec_ptr, slice_ptr, second_portion_len);
                slice_ptr = slice_ptr.add(second_portion_len);
                slice_len -= second_portion_len;
            }

            // Copy any elements remaining from start up to the end of self.vec
            std::ptr::copy_nonoverlapping(self_vec_ptr.add(start), slice_ptr, slice_len);
        }
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
    /// use slice_ring_buf::SliceRB;
    ///
    /// let mut rb = SliceRB::<u32>::from_len(4);
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
        // If slice is longer than self.vec, retreive only the latest portion
        let (slice, start_i) = if slice.len() > self.vec.len() {
            let end_i = start + slice.len() as isize;
            (
                &slice[slice.len() - self.vec.len()..],
                // Find new starting point if slice length has changed
                self.constrain(end_i - self.vec.len() as isize) as usize,
            )
        } else {
            (&slice[..], self.constrain(start) as usize)
        };

        // Safe because self.constrain() is always in range.
        //
        // Memory is created and initialized by a Vec, so it is always valid.
        //
        // Memory cannot overlap because a mutable and immutable reference do not
        // alias.
        unsafe {
            let slice_ptr = slice.as_ptr();
            let self_vec_ptr = self.vec.as_mut_ptr();

            // If the slice is longer than from start_i to the end of self.vec, copy that
            // first portion, then wrap to the beginning and copy the remaining second portion.
            if start_i + slice.len() > self.vec.len() {
                let first_portion_len = self.vec.len() - start_i;
                std::ptr::copy_nonoverlapping(
                    slice_ptr,
                    self_vec_ptr.add(start_i),
                    first_portion_len,
                );

                let second_portion_len = slice.len() - first_portion_len;
                std::ptr::copy_nonoverlapping(
                    slice_ptr.add(first_portion_len),
                    self_vec_ptr,
                    second_portion_len,
                );
            } else {
                // Otherwise, data fits so just copy it
                std::ptr::copy_nonoverlapping(slice_ptr, self_vec_ptr.add(start_i), slice.len());
            }
        }
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
    /// use slice_ring_buf::SliceRB;
    ///
    /// let mut input_rb = SliceRB::<u32>::from_len(4);
    /// input_rb[0] = 1;
    /// input_rb[1] = 2;
    /// input_rb[2] = 3;
    /// input_rb[3] = 4;
    ///
    /// let mut output_rb = SliceRB::<u32>::from_len(4);
    /// // s1 == &[1, 2], s2 == &[]
    /// let (s1, s2) = input_rb.as_slices_len(0, 2);
    /// output_rb.write_latest_2(s1, s2, -3);
    /// assert_eq!(output_rb[0], 0);
    /// assert_eq!(output_rb[1], 1);
    /// assert_eq!(output_rb[2], 2);
    /// assert_eq!(output_rb[3], 0);
    ///
    /// let mut output_rb = SliceRB::<u32>::from_len(2);
    /// // s1 == &[4],  s2 == &[1, 2, 3]
    /// let (s1, s2) = input_rb.as_slices_len(3, 4);
    /// // rb[1] = 4  ->  rb[0] = 1  ->  rb[1] = 2  ->  rb[0] = 3
    /// output_rb.write_latest_2(s1, s2, 1);
    /// assert_eq!(output_rb[0], 3);
    /// assert_eq!(output_rb[1], 2);
    /// ```
    pub fn write_latest_2(&mut self, first: &[T], second: &[T], start: isize) {
        if first.len() + second.len() <= self.vec.len() {
            // All data from both slices need to be copied.
            self.write_latest(first, start);
        } else if second.len() < self.vec.len() {
            // Only data from the end part of first and all of second needs to be copied.
            let first_end_part_len = self.vec.len() - second.len();
            let first_end_part_start = first.len() - first_end_part_len;
            let first_end_part = &first[first_end_part_start..];

            self.write_latest(first_end_part, start + first_end_part_start as isize);
        }
        // else - Only data from second needs to be copied

        self.write_latest(second, start + first.len() as isize);
    }

    /// Returns the length of the ring buffer.
    ///
    /// # Example
    ///
    /// ```
    /// use slice_ring_buf::SliceRB;
    /// let rb = SliceRB::<u32>::from_len(4);
    ///
    /// assert_eq!(rb.len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        self.vec.len()
    }

    /// Returns the allocated capacity of the internal vector.
    ///
    /// Please note this is not the same as the length of the buffer.
    /// For that use `SliceRB::len()`.
    ///
    /// # Example
    ///
    /// ```
    /// use slice_ring_buf::SliceRB;
    /// let rb = SliceRB::<u32>::from_len(4);
    ///
    /// assert!(rb.capacity() >= 4);
    /// ```
    pub fn capacity(&self) -> usize {
        self.vec.capacity()
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
    /// use slice_ring_buf::SliceRB;
    /// let rb = SliceRB::<u32>::from_len(4);
    ///
    /// assert_eq!(rb.constrain(2), 2);
    /// assert_eq!(rb.constrain(4), 0);
    /// assert_eq!(rb.constrain(-3), 1);
    /// assert_eq!(rb.constrain(7), 3);
    /// ```
    #[inline]
    pub fn constrain(&self, i: isize) -> isize {
        if i < 0 || i >= self.len_isize {
            let rem = i % self.len_isize;
            if rem < 0 {
                rem + self.len_isize
            } else {
                rem
            }
        } else {
            i
        }
    }

    /// Returns all the data in the buffer. The starting index will
    /// always be `0`.
    ///
    /// # Example
    ///
    /// ```
    /// use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::from_len(4);
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
    /// use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::from_len(4);
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

    /// Returns the element at the index of type `usize`.
    ///
    /// Please note this does NOT wrap around. This is equivalent to
    /// indexing a normal `Vec`.
    ///
    /// # Example
    ///
    /// ```
    /// use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::from_len(4);
    /// rb[0] = 1;
    /// rb[3] = 4;
    ///
    /// assert_eq!(*rb.raw_at(0), 1);
    /// assert_eq!(*rb.raw_at(3), 4);
    ///
    /// // These will panic!
    /// // assert_eq!(*rb.raw_at(-3), 2);
    /// // assert_eq!(*rb.raw_at(4), 1);
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if `i` is out of bounds of the internal `Vec`.
    #[inline]
    pub fn raw_at(&self, i: usize) -> &T {
        &self.vec[i]
    }

    /// Returns the element at the index of type `usize` as mutable.
    ///
    /// Please note this does NOT wrap around. This is equivalent to
    /// indexing a normal `Vec`.
    ///
    /// # Example
    ///
    /// ```
    /// use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::from_len(4);
    ///
    /// *rb.raw_at_mut(0) = 1;
    /// *rb.raw_at_mut(3) = 4;
    ///
    /// assert_eq!(rb[0], 1);
    /// assert_eq!(rb[3], 4);
    ///
    /// // These will panic!
    /// // *rb.raw_at_mut(-3) = 2;
    /// // *rb.raw_at_mut(4) = 1;
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if `i` is out of bounds of the internal `Vec`.
    #[inline]
    pub fn raw_at_mut(&mut self, i: usize) -> &mut T {
        &mut self.vec[i]
    }

    /// Returns the element at the index of type `usize` while also
    /// constraining the index `i`. This is more efficient
    /// than calling both methods individually.
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
    /// use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::from_len(4);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// let mut i = -3;
    /// assert_eq!(*rb.at(&mut i), 2);
    /// assert_eq!(i, 1);
    /// ```
    #[inline]
    pub fn at(&self, i: &mut isize) -> &T {
        *i = self.constrain(*i);

        // Safe because self.constrain() is always in range.
        //
        // Memory is created and initialized by a Vec, so it is always valid.
        unsafe { &*self.vec.as_ptr().offset(*i) }
    }

    /// Returns the element at the index of type `usize` as mutable while also
    /// constraining the index `i`. This is more efficient
    /// than calling both methods individually.
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
    /// use slice_ring_buf::SliceRB;
    /// let mut rb = SliceRB::<u32>::from_len(4);
    ///
    /// let mut i = -3;
    /// *rb.at_mut(&mut i) = 2;
    ///
    /// assert_eq!(rb[1], 2);
    /// assert_eq!(i, 1);
    /// ```
    #[inline]
    pub fn at_mut(&mut self, i: &mut isize) -> &mut T {
        *i = self.constrain(*i);

        // Safe because self.constrain() is always in range.
        //
        // Memory is created and initialized by a Vec, so it is always valid.
        unsafe { &mut *self.vec.as_mut_ptr().offset(*i) }
    }
}

impl<T: Copy + Clone + Default> std::ops::Index<isize> for SliceRB<T> {
    type Output = T;
    fn index(&self, i: isize) -> &T {
        // Safe because self.constrain() is always in range.
        //
        // Memory is created and initialized by a Vec, so it is always valid.
        unsafe { &*self.vec.as_ptr().offset(self.constrain(i)) }
    }
}

impl<T: Copy + Clone + Default> std::ops::IndexMut<isize> for SliceRB<T> {
    fn index_mut(&mut self, i: isize) -> &mut T {
        // Safe because self.constrain() is always in range.
        //
        // Memory is created and initialized by a Vec, so it is always valid.
        unsafe { &mut *self.vec.as_mut_ptr().offset(self.constrain(i)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_ring_buf_initialize() {
        let ring_buf = SliceRB::<f32>::from_len(3);

        assert_eq!(&ring_buf.vec[..], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn slice_ring_buf_initialize_uninit() {
        unsafe {
            let ring_buf = SliceRB::<f32>::from_len_uninit(3);

            assert_eq!(ring_buf.vec.len(), 3);
        }
    }

    #[test]
    fn slice_ring_buf_clear_set_len() {
        let mut ring_buf = SliceRB::<f32>::from_len(4);
        ring_buf[0] = 1.0;
        ring_buf[1] = 2.0;
        ring_buf[2] = 3.0;
        ring_buf[3] = 4.0;

        ring_buf.clear_set_len(8);
        assert_eq!(ring_buf.vec.as_slice(), &[0.0; 8]);
    }

    #[test]
    fn slice_ring_buf_set_len() {
        let mut ring_buf = SliceRB::<f32>::from_len(4);
        ring_buf[0] = 1.0;
        ring_buf[1] = 2.0;
        ring_buf[2] = 3.0;
        ring_buf[3] = 4.0;

        ring_buf.set_len(1);
        assert_eq!(ring_buf.vec.as_slice(), &[1.0]);

        ring_buf.set_len(4);
        assert_eq!(ring_buf.vec.as_slice(), &[1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn slice_ring_buf_set_len_uninit() {
        let mut ring_buf = SliceRB::<f32>::from_len(4);
        ring_buf[0] = 1.0;
        ring_buf[1] = 2.0;
        ring_buf[2] = 3.0;
        ring_buf[3] = 4.0;

        unsafe {
            ring_buf.set_len_uninit(1);
        }

        assert_eq!(ring_buf.vec.as_slice(), &[1.0]);
        assert_eq!(ring_buf.vec.len(), 1);

        unsafe {
            ring_buf.set_len_uninit(4);
        }

        assert_eq!(ring_buf.vec.len(), 4);
    }

    #[test]
    fn slice_ring_buf_constrain() {
        let ring_buf = SliceRB::<f32>::from_len(4);

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
    fn slice_ring_buf_clear() {
        let mut ring_buf = SliceRB::<f32>::from_len(4);

        ring_buf.write_latest(&[1.0f32, 2.0, 3.0, 4.0], 0);
        assert_eq!(ring_buf.vec.as_slice(), &[1.0, 2.0, 3.0, 4.0]);

        ring_buf.clear();
        assert_eq!(ring_buf.vec.as_slice(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn slice_ring_buf_index() {
        let mut ring_buf = SliceRB::<f32>::from_len(4);
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
        let mut ring_buf = SliceRB::<f32>::from_len(4);
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
        let mut ring_buf = SliceRB::<f32>::from_len(4);
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
        let mut ring_buf = SliceRB::<f32>::from_len(4);
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
        let mut ring_buf = SliceRB::<f32>::from_len(4);

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
        let mut ring_buf = SliceRB::<f32>::from_len(4);

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
        let mut ring_buf = SliceRB::<f32>::from_len(4);
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
        let mut ring_buf = SliceRB::<f32>::from_len(4);
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
        let mut ring_buf = SliceRB::<f32>::from_len(4);
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
        let mut ring_buf = SliceRB::<f32>::from_len(4);
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
        let mut ring_buf = SliceRB::<f32>::from_len(4);
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
