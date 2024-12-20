use core::num::NonZeroUsize;
use core::fmt::Debug;

use super::inner;

/// A ring buffer implementation optimized for working with slices. Note this pretty
/// much does the same thing as [`VecDeque`], but with the added ability to index
/// using negative values, as well as working with buffers allocated on the stack.
/// This struct can be used without the standard library (`#![no_std]`).
///
/// This works the same as [`SliceRB`] except it uses an immutable reference as its
/// data source instead of an internal Vec.
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
/// # use slice_ring_buf::SliceRbRef;
/// let stack_data = [0u32, 1, 2, 3];
/// let rb_ref = SliceRbRef::new(&stack_data);
/// assert_eq!(rb_ref[-3], 1);
///
/// let (s1, s2) = rb_ref.as_slices_len(2, 3);
/// assert_eq!(s1, &[2, 3]);
/// assert_eq!(s2, &[0]);
/// ```
///
/// [`VecDeque`]: https://doc.rust-lang.org/std/collections/struct.VecDeque.html
/// [`SliceRB`]: struct.SliceRB.html
pub struct SliceRbRef<'a, T> {
    data: &'a [T],
}

impl<'a, T> SliceRbRef<'a, T> {
    /// Creates a new [`SliceRbRef`] with the given data.
    ///
    /// # Example
    ///
    /// ```
    /// # use slice_ring_buf::SliceRbRef;
    /// let data = [1u32, 2, 3, 4];
    /// let rb = SliceRbRef::new(&data[..]);
    ///
    /// assert_eq!(rb.len().get(), 4);
    ///
    /// assert_eq!(rb[0], 1);
    /// assert_eq!(rb[1], 2);
    /// assert_eq!(rb[2], 3);
    /// assert_eq!(rb[3], 4);
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if the length of `slice` is `0` or is greater
    /// than `isize::MAX`.
    #[inline]
    pub const fn new(slice: &'a [T]) -> Self {
        assert!(!slice.is_empty());
        assert!(slice.len() <= isize::MAX as usize);

        Self { data: slice }
    }

    /// Creates a new [`SliceRbRef`] with the given data without checking
    /// that the length of the data is greater than `0` and less than or
    /// equal to `isize::MAX`.
    ///
    /// # Example
    ///
    /// ```
    /// # use slice_ring_buf::SliceRbRef;
    /// let data = [1u32, 2, 3, 4];
    /// let rb = unsafe { SliceRbRef::new_unchecked(&data[..]) };
    ///
    /// assert_eq!(rb.len().get(), 4);
    ///
    /// assert_eq!(rb[0], 1);
    /// assert_eq!(rb[1], 2);
    /// assert_eq!(rb[2], 3);
    /// assert_eq!(rb[3], 4);
    /// ```
    ///
    /// # Safety
    ///
    /// The length of `slice` must be greater than `0` and less than or
    /// equal to `isize::MAX`.
    #[inline]
    pub const unsafe fn new_unchecked(slice: &'a [T]) -> Self {
        debug_assert!(!slice.is_empty());
        debug_assert!(slice.len() <= isize::MAX as usize);

        Self { data: slice }
    }

    /// Returns the length of the ring buffer.
    ///
    /// # Example
    ///
    /// ```
    /// # use slice_ring_buf::SliceRbRef;
    /// let data = [0u32; 4];
    /// let rb = SliceRbRef::new(&data[..]);
    ///
    /// assert_eq!(rb.len().get(), 4);
    /// ```
    pub fn len(&self) -> NonZeroUsize {
        // SAFETY:
        // * All constructors ensure that the length is greater than `0`.
        unsafe { NonZeroUsize::new_unchecked(self.data.len()) }
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
    /// `SliceRbRef::at(&mut i)` over `SliceRbRef[i]` to reduce the number of
    /// modulo operations to perform.
    ///
    /// # Example
    ///
    /// ```
    /// # use slice_ring_buf::SliceRbRef;
    /// let data = [0u32; 4];
    /// let rb = SliceRbRef::new(&data[..]);
    ///
    /// assert_eq!(rb.constrain(2), 2);
    /// assert_eq!(rb.constrain(4), 0);
    /// assert_eq!(rb.constrain(-3), 1);
    /// assert_eq!(rb.constrain(7), 3);
    /// ```
    pub fn constrain(&self, i: isize) -> isize {
        inner::constrain(i, self.data.len() as isize)
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
    /// # use slice_ring_buf::SliceRbRef;
    /// let data = [1u32, 2, 3, 4];
    /// let rb = SliceRbRef::new(&data[..]);
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
        // The constructors ensure that `self.data.len() > 0 && self.data.len() <= isize::MAX`.
        unsafe { inner::as_slices(start, self.data) }
    }

    /// Returns two slices of data in the ring buffer
    /// starting at the index `start` and with length `len`.
    ///
    /// * `start` - The starting index
    /// * `len` - The length of data to read. If `len` is greater than the
    /// capacity of the ring buffer, then that capacity will be used instead.
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
    /// # use slice_ring_buf::SliceRbRef;
    /// let data = [1u32, 2, 3, 4];
    /// let rb = SliceRbRef::new(&data[..]);
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
        // The constructors ensure that `self.data.len() > 0 && self.data.len() <= isize::MAX`.
        unsafe { inner::as_slices_len(start, len, self.data) }
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
    /// # use slice_ring_buf::SliceRbRef;
    /// let data = [1u32, 2, 3, 4];
    /// let rb = SliceRbRef::new(&data[..]);
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
        // The constructors ensure that `self.data.len() > 0 && self.data.len() <= isize::MAX`.
        unsafe { inner::as_slices_latest(start, len, self.data) }
    }

    /// Returns all the data in the buffer. The starting index will
    /// always be `0`.
    ///
    /// # Example
    ///
    /// ```
    /// # use slice_ring_buf::SliceRbRef;
    /// let data = [1u32, 2, 3, 4];
    /// let rb = SliceRbRef::new(&data[..]);
    ///
    /// let raw_data = rb.raw_data();
    /// assert_eq!(raw_data, &[1u32, 2, 3, 4]);
    /// ```
    pub fn raw_data(&self) -> &[T] {
        self.data
    }

    /// Returns an immutable reference the element at the index of type `isize`.
    ///
    /// # Performance
    ///
    /// Prefer to manipulate data in bulk with methods that return slices. If you
    /// need to index multiple elements one at a time, prefer to use
    /// this over `SliceRbRef[i]` to reduce the number of
    /// modulo operations to perform.
    ///
    /// # Example
    ///
    /// ```
    /// # use slice_ring_buf::SliceRbRef;
    /// let data = [1u32, 2, 3, 4];
    /// let rb = SliceRbRef::new(&data[..]);
    ///
    /// assert_eq!(*rb.get(-3), 2);
    /// ```
    #[inline]
    pub fn get(&self, i: isize) -> &T {
        // SAFETY:
        // The constructors ensure that `self.data.len() > 0 && self.data.len() <= isize::MAX`.
        unsafe { inner::get(i, self.data) }
    }

    /// Returns an immutable reference to the element at the index of type `isize`
    /// while also constraining the index `i`. This is more efficient than calling
    /// both methods individually.
    ///
    /// # Performance
    ///
    /// Prefer to manipulate data in bulk with methods that return slices. If you
    /// need to index multiple elements one at a time, prefer to use
    /// this over `SliceRbRef[i]` to reduce the number of
    /// modulo operations to perform.
    ///
    /// # Example
    ///
    /// ```
    /// # use slice_ring_buf::SliceRbRef;
    /// let data = [1u32, 2, 3, 4];
    /// let rb = SliceRbRef::new(&data[..]);
    ///
    /// let mut i = -3;
    /// assert_eq!(*rb.constrain_and_get(&mut i), 2);
    /// assert_eq!(i, 1);
    /// ```
    #[inline]
    pub fn constrain_and_get(&self, i: &mut isize) -> &T {
        // SAFETY:
        // The constructors ensure that `self.data.len() > 0 && self.data.len() <= isize::MAX`.
        unsafe { inner::constrain_and_get(i, self.data) }
    }
}

impl<'a, T: Clone + Copy> SliceRbRef<'a, T> {
    /// Copies the data from the ring buffer starting from the index `start`
    /// into the given slice. If the length of `slice` is larger than the
    /// capacity of the ring buffer, then the data will be reapeated until
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
    /// # use slice_ring_buf::SliceRbRef;
    /// let data = [1u32, 2, 3, 4];
    /// let rb = SliceRbRef::new(&data[..]);
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
        // The constructors ensure that `self.data.len() > 0 && self.data.len() <= isize::MAX`.
        unsafe { inner::read_into(slice, start, self.data) }
    }
}

/// A ring buffer implementation optimized for working with slices. Note this pretty
/// much does the same thing as [`VecDeque`], but with the added ability to index
/// using negative values, as well as working with buffers allocated on the stack.
/// This struct can be used without the standard library (`#![no_std]`).
///
/// This works the same as [`SliceRB`] except it uses a mutable reference as its
/// data source instead of an internal Vec.
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
/// # use slice_ring_buf::SliceRbRefMut;
/// let mut stack_data = [0u32, 1, 2, 3];
/// let mut rb_ref = SliceRbRefMut::new(&mut stack_data);
///
/// rb_ref[-4] = 5;
///
/// let (s1, s2) = rb_ref.as_slices_len(2, 3);
/// assert_eq!(s1, &[2, 3]);
/// assert_eq!(s2, &[5]);
/// ```
///
/// [`VecDeque`]: https://doc.rust-lang.org/std/collections/struct.VecDeque.html
/// [`SliceRB`]: struct.SliceRB.html
pub struct SliceRbRefMut<'a, T> {
    data: &'a mut [T],
}

impl<'a, T> SliceRbRefMut<'a, T> {
    /// Creates a new [`SliceRbRefMut`] with the given data.
    ///
    /// # Example
    ///
    /// ```
    /// # use slice_ring_buf::SliceRbRefMut;
    /// let mut data = [1u32, 2, 3, 4];
    /// let rb = SliceRbRefMut::new(&mut data[..]);
    ///
    /// assert_eq!(rb.len(), 4);
    ///
    /// assert_eq!(rb[0], 1);
    /// assert_eq!(rb[1], 2);
    /// assert_eq!(rb[2], 3);
    /// assert_eq!(rb[3], 4);
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if the length of `slice` is `0` or is greater
    /// than `isize::MAX`.
    #[inline]
    pub const fn new(slice: &'a mut [T]) -> Self {
        assert!(!slice.is_empty());
        assert!(slice.len() <= isize::MAX as usize);

        Self { data: slice }
    }

    /// Creates a new [`SliceRbRefMut`] with the given data without checking
    /// that the length of the data is greater than `0` and less than or
    /// equal to `isize::MAX`.
    ///
    /// # Example
    ///
    /// ```
    /// # use slice_ring_buf::SliceRbRefMut;
    /// let mut data = [1u32, 2, 3, 4];
    /// let rb = unsafe { SliceRbRefMut::new_unchecked(&mut data[..]) };
    ///
    /// assert_eq!(rb.len(), 4);
    ///
    /// assert_eq!(rb[0], 1);
    /// assert_eq!(rb[1], 2);
    /// assert_eq!(rb[2], 3);
    /// assert_eq!(rb[3], 4);
    /// ```
    ///
    /// # Safety
    ///
    /// The length of `slice` must be greater than `0` and less than or
    /// equal to `isize::MAX`.
    #[inline]
    pub const unsafe fn new_unchecked(slice: &'a mut [T]) -> Self {
        debug_assert!(!slice.is_empty());
        debug_assert!(slice.len() <= isize::MAX as usize);

        Self { data: slice }
    }

    /// Returns the length of the ring buffer.
    ///
    /// # Example
    ///
    /// ```
    /// # use slice_ring_buf::SliceRbRefMut;
    /// let mut data = [0u32; 4];
    /// let mut rb = SliceRbRefMut::new(&mut data[..]);
    ///
    /// assert_eq!(rb.len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        self.data.len()
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
    /// `SliceRbRef::at(&mut i)` over `SliceRbRef[i]` to reduce the number of
    /// modulo operations to perform.
    ///
    /// # Example
    ///
    /// ```
    /// # use slice_ring_buf::SliceRbRefMut;
    /// let mut data = [0u32; 4];
    /// let mut rb = SliceRbRefMut::new(&mut data[..]);
    ///
    /// assert_eq!(rb.constrain(2), 2);
    /// assert_eq!(rb.constrain(4), 0);
    /// assert_eq!(rb.constrain(-3), 1);
    /// assert_eq!(rb.constrain(7), 3);
    /// ```
    pub fn constrain(&self, i: isize) -> isize {
        inner::constrain(i, self.data.len() as isize)
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
    /// # use slice_ring_buf::SliceRbRefMut;
    /// let mut data = [1u32, 2, 3, 4];
    /// let mut rb = SliceRbRefMut::new(&mut data[..]);
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
        // The constructors ensure that `self.data.len() > 0 && self.data.len() <= isize::MAX`.
        unsafe { inner::as_slices(start, self.data) }
    }

    /// Returns two slices of data in the ring buffer
    /// starting at the index `start` and with length `len`.
    ///
    /// * `start` - The starting index
    /// * `len` - The length of data to read. If `len` is greater than the
    /// capacity of the ring buffer, then that capacity will be used instead.
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
    /// # use slice_ring_buf::SliceRbRefMut;
    /// let mut data = [1u32, 2, 3, 4];
    /// let mut rb = SliceRbRefMut::new(&mut data[..]);
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
        // The constructors ensure that `self.data.len() > 0 && self.data.len() <= isize::MAX`.
        unsafe { inner::as_slices_len(start, len, self.data) }
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
    /// # use slice_ring_buf::SliceRbRefMut;
    /// let mut data = [1u32, 2, 3, 4];
    /// let rb = SliceRbRefMut::new(&mut data[..]);
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
        // The constructors ensure that `self.data.len() > 0 && self.data.len() <= isize::MAX`.
        unsafe { inner::as_slices_latest(start, len, self.data) }
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
    /// # use slice_ring_buf::SliceRbRefMut;
    /// let mut data = [1u32, 2, 3, 4];
    /// let mut rb = SliceRbRefMut::new(&mut data[..]);
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
        // The constructors ensure that `self.data.len() > 0 && self.data.len() <= isize::MAX`.
        unsafe { inner::as_mut_slices(start, self.data) }
    }

    /// Returns two mutable slices of data in the ring buffer
    /// starting at the index `start` and with length `len`.
    ///
    /// * `start` - The starting index
    /// * `len` - The length of data to read. If `len` is greater than the
    /// capacity of the ring buffer, then that capacity will be used instead.
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
    /// # use slice_ring_buf::SliceRbRefMut;
    /// let mut data = [1u32, 2, 3, 4];
    /// let mut rb = SliceRbRefMut::new(&mut data[..]);
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
        // The constructors ensure that `self.data.len() > 0 && self.data.len() <= isize::MAX`.
        unsafe { inner::as_mut_slices_len(start, len, self.data) }
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
    /// # use slice_ring_buf::SliceRbRefMut;
    /// let mut data = [1u32, 2, 3, 4];
    /// let mut rb = SliceRbRefMut::new(&mut data[..]);
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
        // The constructors ensure that `self.data.len() > 0 && self.data.len() <= isize::MAX`.
        unsafe { inner::as_mut_slices_latest(start, len, self.data) }
    }

    /// Returns all the data in the buffer. The starting index will
    /// always be `0`.
    ///
    /// # Example
    ///
    /// ```
    /// # use slice_ring_buf::SliceRbRefMut;
    /// let mut data = [1u32, 2, 3, 4];
    /// let mut rb = SliceRbRefMut::new(&mut data[..]);
    ///
    /// let raw_data = rb.raw_data();
    /// assert_eq!(raw_data, &[1u32, 2, 3, 4]);
    /// ```
    pub fn raw_data(&self) -> &[T] {
        self.data
    }

    /// Returns all the data in the buffer as mutable. The starting
    /// index will always be `0`.
    ///
    /// # Example
    ///
    /// ```
    /// # use slice_ring_buf::SliceRbRefMut;
    /// let mut data = [1u32, 2, 3, 4];
    /// let mut rb = SliceRbRefMut::new(&mut data[..]);
    ///
    /// let raw_data = rb.raw_data_mut();
    /// assert_eq!(raw_data, &mut [1u32, 2, 3, 4]);
    /// ```
    pub fn raw_data_mut(&mut self) -> &mut [T] {
        self.data
    }

    /// Returns an immutable reference the element at the index of type `isize`.
    ///
    /// # Performance
    ///
    /// Prefer to manipulate data in bulk with methods that return slices. If you
    /// need to index multiple elements one at a time, prefer to use
    /// this over `SliceRbRef[i]` to reduce the number of
    /// modulo operations to perform.
    ///
    /// # Example
    ///
    /// ```
    /// # use slice_ring_buf::SliceRbRefMut;
    /// let mut data = [1u32, 2, 3, 4];
    /// let mut rb = SliceRbRefMut::new(&mut data[..]);
    ///
    /// assert_eq!(*rb.get(-3), 2);
    /// ```
    #[inline]
    pub fn get(&self, i: isize) -> &T {
        // SAFETY:
        // The constructors ensure that `self.data.len() > 0 && self.data.len() <= isize::MAX`.
        unsafe { inner::get(i, self.data) }
    }

    /// Returns a mutable reference the element at the index of type `isize`.
    ///
    /// # Performance
    ///
    /// Prefer to manipulate data in bulk with methods that return slices. If you
    /// need to index multiple elements one at a time, prefer to use
    /// this over `SliceRbRef[i]` to reduce the number of
    /// modulo operations to perform.
    ///
    /// # Example
    ///
    /// ```
    /// # use slice_ring_buf::SliceRbRefMut;
    /// let mut data = [1u32, 2, 3, 4];
    /// let mut rb = SliceRbRefMut::new(&mut data[..]);
    ///
    /// *rb.get_mut(-3) = 5;
    ///
    /// assert_eq!(*rb.get(-3), 5);
    /// ```
    #[inline]
    pub fn get_mut(&mut self, i: isize) -> &mut T {
        // SAFETY:
        // The constructors ensure that `self.data.len() > 0 && self.data.len() <= isize::MAX`.
        unsafe { inner::get_mut(i, self.data) }
    }

    /// Returns an immutable reference to the element at the index of type `isize`
    /// while also constraining the index `i`. This is more efficient than calling
    /// both methods individually.
    ///
    /// # Performance
    ///
    /// Prefer to manipulate data in bulk with methods that return slices. If you
    /// need to index multiple elements one at a time, prefer to use
    /// this over `SliceRbRef[i]` to reduce the number of
    /// modulo operations to perform.
    ///
    /// # Example
    ///
    /// ```
    /// # use slice_ring_buf::SliceRbRefMut;
    /// let mut data = [1u32, 2, 3, 4];
    /// let mut rb = SliceRbRefMut::new(&mut data[..]);
    ///
    /// let mut i = -3;
    /// assert_eq!(*rb.constrain_and_get(&mut i), 2);
    /// assert_eq!(i, 1);
    /// ```
    #[inline]
    pub fn constrain_and_get(&self, i: &mut isize) -> &T {
        // SAFETY:
        // The constructors ensure that `self.data.len() > 0 && self.data.len() <= isize::MAX`.
        unsafe { inner::constrain_and_get(i, self.data) }
    }

    /// Returns a mutable reference to the element at the index of type `isize` as
    /// mutable while also constraining the index `i`. This is more efficient than
    /// calling both methods individually.
    ///
    /// # Performance
    ///
    /// Prefer to manipulate data in bulk with methods that return slices. If you
    /// need to index multiple elements one at a time, prefer to use
    /// this over `SliceRbRef[i]` to reduce the number of
    /// modulo operations to perform.
    ///
    /// # Example
    ///
    /// ```
    /// # use slice_ring_buf::SliceRbRefMut;
    /// let mut data = [0u32; 4];
    /// let mut rb = SliceRbRefMut::new(&mut data[..]);
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
        // The constructors ensure that `self.data.len() > 0 && self.data.len() <= isize::MAX`.
        unsafe { inner::constrain_and_get_mut(i, self.data) }
    }
}

impl<'a, T: Clone + Copy> SliceRbRefMut<'a, T> {
    /// Copies the data from the ring buffer starting from the index `start`
    /// into the given slice. If the length of `slice` is larger than the
    /// capacity of the ring buffer, then the data will be reapeated until
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
    /// # use slice_ring_buf::SliceRbRefMut;
    /// let mut data = [1u32, 2, 3, 4];
    /// let mut rb = SliceRbRefMut::new(&mut data[..]);
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
        // The constructors ensure that `self.data.len() > 0 && self.data.len() <= isize::MAX`.
        unsafe { inner::read_into(slice, start, self.data) }
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
    /// # use slice_ring_buf::SliceRbRefMut;
    /// let mut data = [0u32; 4];
    /// let mut rb = SliceRbRefMut::new(&mut data[..]);
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
        // The constructors ensure that `self.data.len() > 0 && self.data.len() <= isize::MAX`.
        unsafe { inner::write_latest(slice, start, self.data) }
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
    /// # use slice_ring_buf::{SliceRB, SliceRbRefMut};
    /// let mut input_rb = SliceRB::<u32>::new(NonZeroUsize::new(4).unwrap(), 0);
    /// input_rb[0] = 1;
    /// input_rb[1] = 2;
    /// input_rb[2] = 3;
    /// input_rb[3] = 4;
    ///
    /// let mut output_data = [0u32; 4];
    /// let mut output_rb = SliceRbRefMut::new(&mut output_data[..]);
    /// // s1 == &[1, 2], s2 == &[]
    /// let (s1, s2) = input_rb.as_slices_len(0, 2);
    /// output_rb.write_latest_2(s1, s2, -3);
    /// assert_eq!(output_rb[0], 0);
    /// assert_eq!(output_rb[1], 1);
    /// assert_eq!(output_rb[2], 2);
    /// assert_eq!(output_rb[3], 0);
    ///
    /// let mut output_data = [0u32; 2];
    /// let mut output_rb = SliceRbRefMut::new(&mut output_data[..]);
    /// // s1 == &[4],  s2 == &[1, 2, 3]
    /// let (s1, s2) = input_rb.as_slices_len(3, 4);
    /// // rb[1] = 4  ->  rb[0] = 1  ->  rb[1] = 2  ->  rb[0] = 3
    /// output_rb.write_latest_2(s1, s2, 1);
    /// assert_eq!(output_rb[0], 3);
    /// assert_eq!(output_rb[1], 2);
    /// ```
    pub fn write_latest_2(&mut self, first: &[T], second: &[T], start: isize) {
        // SAFETY:
        // The constructors ensure that `self.data.len() > 0 && self.data.len() <= isize::MAX`.
        unsafe { inner::write_latest_2(first, second, start, self.data) }
    }
}

impl<'a, T> Into<SliceRbRef<'a, T>> for SliceRbRefMut<'a, T> {
    fn into(self) -> SliceRbRef<'a, T> {
        SliceRbRef { data: self.data }
    }
}

impl<'a, T> core::ops::Index<isize> for SliceRbRef<'a, T> {
    type Output = T;
    fn index(&self, i: isize) -> &T {
        self.get(i)
    }
}

impl<'a, T> core::ops::Index<isize> for SliceRbRefMut<'a, T> {
    type Output = T;
    fn index(&self, i: isize) -> &T {
        self.get(i)
    }
}

impl<'a, T> core::ops::IndexMut<isize> for SliceRbRefMut<'a, T> {
    fn index_mut(&mut self, i: isize) -> &mut T {
        self.get_mut(i)
    }
}

impl<'a, T: Debug> Debug for SliceRbRef<'a, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut f = f.debug_struct("SliceRbRef");
        f.field("data", &self.data);
        f.finish()
    }
}

impl<'a, T: Debug> Debug for SliceRbRefMut<'a, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut f = f.debug_struct("SliceRbRefMut");
        f.field("data", &self.data);
        f.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_ring_buf_ref_initialize() {
        let mut data = [0.0; 4];
        let ring_buf = SliceRbRef::new(&mut data);

        assert_eq!(&ring_buf.data[..], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn slice_ring_buf_ref_constrain() {
        let mut data = [0.0; 4];
        let ring_buf = SliceRbRef::new(&mut data);

        assert_eq!(&ring_buf.data[..], &[0.0, 0.0, 0.0, 0.0]);

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
    fn slice_ring_buf_ref_index() {
        let mut data = [0.0f32, 1.0, 2.0, 3.0];
        let ring_buf = SliceRbRef::new(&mut data);

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
    fn slice_ring_buf_ref_index_mut() {
        let mut data = [0.0f32, 1.0, 2.0, 3.0];
        let mut ring_buf = SliceRbRefMut::new(&mut data);

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
    fn slice_ring_buf_ref_as_slices() {
        let mut data = [1.0f32, 2.0, 3.0, 4.0];
        let ring_buf = SliceRbRef::new(&mut data);

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
    fn slice_ring_buf_ref_as_mut_slices() {
        let mut data = [1.0f32, 2.0, 3.0, 4.0];
        let mut ring_buf = SliceRbRefMut::new(&mut data);

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

    #[repr(C, align(32))]
    struct Aligned324([f32; 4]);

    #[test]
    fn slice_ring_buf_ref_write_latest_2() {
        let mut data = Aligned324([0.0f32; 4]);
        let mut ring_buf = SliceRbRefMut::new(&mut data.0);

        ring_buf.write_latest_2(&[], &[0.0, 1.0, 2.0, 3.0, 4.0], 1);
        assert_eq!(ring_buf.data, &[3.0, 4.0, 1.0, 2.0]);
        ring_buf.write_latest_2(&[-1.0], &[0.0, 1.0, 2.0, 3.0, 4.0], 1);
        assert_eq!(ring_buf.data, &[2.0, 3.0, 4.0, 1.0]);
        ring_buf.write_latest_2(&[-2.0, -1.0], &[0.0, 1.0, 2.0, 3.0, 4.0], 1);
        assert_eq!(ring_buf.data, &[1.0, 2.0, 3.0, 4.0]);
        ring_buf.write_latest_2(&[-2.0, -1.0], &[0.0, 1.0], 3);
        assert_eq!(ring_buf.data, &[-1.0, 0.0, 1.0, -2.0]);
        ring_buf.write_latest_2(&[0.0, 1.0], &[2.0], 3);
        assert_eq!(ring_buf.data, &[1.0, 2.0, 1.0, 0.0]);
        ring_buf.write_latest_2(&[1.0, 2.0, 3.0, 4.0], &[], 0);
        assert_eq!(ring_buf.data, &[1.0, 2.0, 3.0, 4.0]);
        ring_buf.write_latest_2(&[1.0, 2.0], &[], 2);
        assert_eq!(ring_buf.data, &[1.0, 2.0, 1.0, 2.0]);
        ring_buf.write_latest_2(&[], &[], 2);
        assert_eq!(ring_buf.data, &[1.0, 2.0, 1.0, 2.0]);
        ring_buf.write_latest_2(&[1.0, 2.0, 3.0, 4.0, 5.0], &[], 1);
        assert_eq!(ring_buf.data, &[4.0, 5.0, 2.0, 3.0]);
        ring_buf.write_latest_2(&[1.0, 2.0, 3.0, 4.0, 5.0], &[6.0], 2);
        assert_eq!(ring_buf.data, &[3.0, 4.0, 5.0, 6.0]);
        ring_buf.write_latest_2(&[1.0, 2.0, 3.0, 4.0, 5.0], &[6.0, 7.0], 2);
        assert_eq!(ring_buf.data, &[7.0, 4.0, 5.0, 6.0]);
        ring_buf.write_latest_2(&[1.0, 2.0, 3.0, 4.0, 5.0], &[6.0, 7.0, 8.0, 9.0, 10.0], 3);
        assert_eq!(ring_buf.data, &[10.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn slice_ring_buf_ref_write_latest() {
        let mut data = Aligned324([0.0f32; 4]);
        let mut ring_buf = SliceRbRefMut::new(&mut data.0);

        let input = [0.0f32, 1.0, 2.0, 3.0];

        ring_buf.write_latest(&input, 0);
        assert_eq!(ring_buf.data, &[0.0, 1.0, 2.0, 3.0]);
        ring_buf.write_latest(&input, 1);
        assert_eq!(ring_buf.data, &[3.0, 0.0, 1.0, 2.0]);
        ring_buf.write_latest(&input, 2);
        assert_eq!(ring_buf.data, &[2.0, 3.0, 0.0, 1.0]);
        ring_buf.write_latest(&input, 3);
        assert_eq!(ring_buf.data, &[1.0, 2.0, 3.0, 0.0]);
        ring_buf.write_latest(&input, 4);
        assert_eq!(ring_buf.data, &[0.0, 1.0, 2.0, 3.0]);

        let input = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        ring_buf.write_latest(&input, 0);
        assert_eq!(ring_buf.data, &[4.0, 5.0, 6.0, 7.0]);
        ring_buf.write_latest(&input, 1);
        assert_eq!(ring_buf.data, &[7.0, 4.0, 5.0, 6.0]);
        ring_buf.write_latest(&input, 2);
        assert_eq!(ring_buf.data, &[6.0, 7.0, 4.0, 5.0]);
        ring_buf.write_latest(&input, 3);
        assert_eq!(ring_buf.data, &[5.0, 6.0, 7.0, 4.0]);
        ring_buf.write_latest(&input, 4);
        assert_eq!(ring_buf.data, &[4.0, 5.0, 6.0, 7.0]);

        let input = [0.0f32, 1.0];

        ring_buf.write_latest(&input, 0);
        assert_eq!(ring_buf.data, &[0.0, 1.0, 6.0, 7.0]);
        ring_buf.write_latest(&input, 1);
        assert_eq!(ring_buf.data, &[0.0, 0.0, 1.0, 7.0]);
        ring_buf.write_latest(&input, 2);
        assert_eq!(ring_buf.data, &[0.0, 0.0, 0.0, 1.0]);
        ring_buf.write_latest(&input, 3);
        assert_eq!(ring_buf.data, &[1.0, 0.0, 0.0, 0.0]);
        ring_buf.write_latest(&input, 4);
        assert_eq!(ring_buf.data, &[0.0, 1.0, 0.0, 0.0]);

        let aligned_input = Aligned1([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.data, &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned2([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.data, &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned4([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.data, &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned8([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.data, &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned16([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.data, &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned32([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.data, &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned64([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.data, &[12.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn slice_ring_buf_ref_as_slices_len() {
        let mut data = [0.0f32, 1.0, 2.0, 3.0];
        let ring_buf = SliceRbRef::new(&mut data);

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
    fn slice_ring_buf_ref_as_slices_latest() {
        let mut data = [0.0f32, 1.0, 2.0, 3.0];
        let ring_buf = SliceRbRefMut::new(&mut data);

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
    fn slice_ring_buf_ref_as_mut_slices_len() {
        let mut data = [0.0f32, 1.0, 2.0, 3.0];
        let mut ring_buf = SliceRbRefMut::new(&mut data);

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
    fn slice_ring_buf_ref_as_mut_slices_latest() {
        let mut data = [0.0f32, 1.0, 2.0, 3.0];
        let mut ring_buf = SliceRbRefMut::new(&mut data);

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
    fn slice_ring_buf_ref_read_into() {
        let mut data = Aligned324([0.0f32, 1.0, 2.0, 3.0]);
        let ring_buf = SliceRbRef::new(&mut data.0);

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
