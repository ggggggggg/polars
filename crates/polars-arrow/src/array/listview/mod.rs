use super::specification::try_check_unordered_offset_length_pairs_bounds;
use super::{new_empty_array, Array, Splitable};
use crate::bitmap::Bitmap;
use crate::datatypes::{ArrowDataType, Field};
use crate::offset::{Offset, Offsets, OffsetsBuffer};

#[cfg(feature = "arrow_rs")]
mod data;
mod ffi;
pub(super) mod fmt;
mod iterator;
pub use iterator::*;
mod mutable;
pub use mutable::*;
use polars_error::{polars_bail, PolarsResult};

// macros copied from array/mod.rs because I can't figure out visibility issues 

// macro implementing `with_validity` and `set_validity`
macro_rules! impl_mut_validity {
    () => {
        /// Returns this array with a new validity.
        /// # Panic
        /// Panics iff `validity.len() != self.len()`.
        #[must_use]
        #[inline]
        pub fn with_validity(mut self, validity: Option<Bitmap>) -> Self {
            self.set_validity(validity);
            self
        }

        /// Sets the validity of this array.
        /// # Panics
        /// This function panics iff `values.len() != self.len()`.
        #[inline]
        pub fn set_validity(&mut self, validity: Option<Bitmap>) {
            if matches!(&validity, Some(bitmap) if bitmap.len() != self.len()) {
                panic!("validity must be equal to the array's length")
            }
            self.validity = validity;
        }

        /// Takes the validity of this array, leaving it without a validity mask.
        #[inline]
        pub fn take_validity(&mut self) -> Option<Bitmap> {
            self.validity.take()
        }
    }
}

// macro implementing `sliced` and `sliced_unchecked`
macro_rules! impl_sliced {
    () => {
        /// Returns this array sliced.
        /// # Implementation
        /// This function is `O(1)`.
        /// # Panics
        /// iff `offset + length > self.len()`.
        #[inline]
        #[must_use]
        pub fn sliced(self, offset: usize, length: usize) -> Self {
            assert!(
                offset + length <= self.len(),
                "the offset of the new Buffer cannot exceed the existing length"
            );
            unsafe { Self::sliced_unchecked(self, offset, length) }
        }

        /// Returns this array sliced.
        /// # Implementation
        /// This function is `O(1)`.
        ///
        /// # Safety
        /// The caller must ensure that `offset + length <= self.len()`.
        #[inline]
        #[must_use]
        pub unsafe fn sliced_unchecked(mut self, offset: usize, length: usize) -> Self {
            Self::slice_unchecked(&mut self, offset, length);
            self
        }
    };
}

// macro implementing `boxed` and `arced`
macro_rules! impl_into_array {
    () => {
        /// Boxes this array into a [`Box<dyn Array>`].
        pub fn boxed(self) -> Box<dyn Array> {
            Box::new(self)
        }

        /// Arcs this array into a [`std::sync::Arc<dyn Array>`].
        pub fn arced(self) -> std::sync::Arc<dyn Array> {
            std::sync::Arc::new(self)
        }
    };
}

// macro implementing common methods of trait `Array`
macro_rules! impl_common_array {
    () => {
        #[inline]
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        #[inline]
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }

        #[inline]
        fn len(&self) -> usize {
            self.len()
        }

        #[inline]
        fn data_type(&self) -> &ArrowDataType {
            &self.data_type
        }

        #[inline]
        fn split_at_boxed(&self, offset: usize) -> (Box<dyn Array>, Box<dyn Array>) {
            let (lhs, rhs) = $crate::array::Splitable::split_at(self, offset);
            (Box::new(lhs), Box::new(rhs))
        }

        #[inline]
        unsafe fn split_at_boxed_unchecked(
            &self,
            offset: usize,
        ) -> (Box<dyn Array>, Box<dyn Array>) {
            let (lhs, rhs) = unsafe { $crate::array::Splitable::split_at_unchecked(self, offset) };
            (Box::new(lhs), Box::new(rhs))
        }

        #[inline]
        fn slice(&mut self, offset: usize, length: usize) {
            self.slice(offset, length);
        }

        #[inline]
        unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
            self.slice_unchecked(offset, length);
        }

        #[inline]
        fn to_boxed(&self) -> Box<dyn Array> {
            Box::new(self.clone())
        }
    };
}


/// An [`Array`] semantically equivalent to `Vec<Option<Vec<Option<T>>>>` with Arrow's in-memory.
#[derive(Clone)]
pub struct ListViewArray<O: Offset> {
    data_type: ArrowDataType,
    offsets: OffsetsBuffer<O>,
    lengths: OffsetsBuffer<O>,
    values: Box<dyn Array>,
    validity: Option<Bitmap>,
}



impl<O: Offset> ListViewArray<O> {
    /// Creates a new [`ListViewArray`].
    ///
    /// # Errors
    /// This function returns an error iff:
    /// * not all 0 <= offsets[i] <= values.len()
    /// * not all 0 <= offsets[i] + lengths[i] <= values.len()
    /// * the validity's length is not equal to `offsets.len()`.
    /// * The `data_type`'s [`crate::datatypes::PhysicalType`] is not equal to either [`crate::datatypes::PhysicalType::ListView`] or [`crate::datatypes::PhysicalType::LargeListView`].
    /// * The `data_type`'s inner field's data type is not equal to `values.data_type`.
    /// # Implementation
    /// This function is `O(1)`
    pub fn try_new(
        data_type: ArrowDataType,
        offsets: OffsetsBuffer<O>,
        lengths: OffsetsBuffer<O>,
        values: Box<dyn Array>,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        try_check_unordered_offset_length_pairs_bounds(&offsets, &lengths, values.len())?;

        if validity
            .as_ref()
            .map_or(false, |validity| validity.len() != offsets.len_proxy())
        {
            polars_bail!(ComputeError: "validity mask length must match the number of values")
        }

        let child_data_type = Self::try_get_child(&data_type)?.data_type();
        let values_data_type = values.data_type();
        if child_data_type != values_data_type {
            polars_bail!(ComputeError: "ListViewArray's child's DataType must match. However, the expected DataType is {child_data_type:?} while it got {values_data_type:?}.");
        }

        Ok(Self {
            data_type,
            offsets,
            lengths,
            values,
            validity,
        })
    }

    /// Creates a new [`ListViewArray`].
    ///
    /// # Panics
    /// This function panics iff:
    /// * not all 0 <= offsets[i] <= values.len()
    /// * not all 0 <= offsets[i] + lengths[i] <= values.len()
    /// * the validity's length is not equal to `offsets.len()`.
    /// * The `data_type`'s [`crate::datatypes::PhysicalType`] is not equal to either [`crate::datatypes::PhysicalType::List`] or [`crate::datatypes::PhysicalType::LargeList`].
    /// * The `data_type`'s inner field's data type is not equal to `values.data_type`.
    /// # Implementation
    /// This function is `O(1)`
    pub fn new(
        data_type: ArrowDataType,
        offsets: OffsetsBuffer<O>,
        lengths: OffsetsBuffer<O>,
        values: Box<dyn Array>,
        validity: Option<Bitmap>,
    ) -> Self {
        Self::try_new(data_type, offsets, lengths,values, validity).unwrap()
    }

    /// Returns a new empty [`ListViewArray`].
    pub fn new_empty(data_type: ArrowDataType) -> Self {
        let values = new_empty_array(Self::get_child_type(&data_type).clone());
        Self::new(data_type, OffsetsBuffer::default(), OffsetsBuffer::default(), values, None)
    }

    /// Returns a new null [`ListViewArray`].
    #[inline]
    pub fn new_null(data_type: ArrowDataType, length: usize) -> Self {
        let child = Self::get_child_type(&data_type).clone();
        Self::new(
            data_type,
            Offsets::new_zeroed(length).into(),
            Offsets::new_zeroed(length).into(),
            new_empty_array(child), 
            Some(Bitmap::new_zeroed(length)),
        )
    }
}

impl<O: Offset> ListViewArray<O> {
    /// Slices this [`ListViewArray`].
    /// # Panics
    /// panics iff `offset + length > self.len()`
    pub fn slice(&mut self, offset: usize, length: usize) {
        assert!(
            offset + length <= self.len(),
            "the offset of the new Buffer cannot exceed the existing length"
        );
        unsafe { self.slice_unchecked(offset, length) }
    }

    /// Slices this [`ListViewArray`].
    ///
    /// # Safety
    /// The caller must ensure that `offset + length < self.len()`.
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        self.validity = self
            .validity
            .take()
            .map(|bitmap| bitmap.sliced_unchecked(offset, length))
            .filter(|bitmap| bitmap.unset_bits() > 0);
        self.offsets.slice_unchecked(offset, length + 1);
    }

    impl_sliced!();
    impl_mut_validity!();
    impl_into_array!();
}

// Accessors
impl<O: Offset> ListViewArray<O> {
    /// Returns the length of this array
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len_proxy()
    }

    /// Returns the element at index `i`
    /// # Panic
    /// Panics iff `i >= self.len()`
    #[inline]
    pub fn value(&self, i: usize) -> Box<dyn Array> {
        assert!(i < self.len());
        // SAFETY: invariant of this function
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the element at index `i` as &str
    ///
    /// # Safety
    /// Assumes that the `i < self.len`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> Box<dyn Array> {
        // SAFETY: the invariant of the function
        let (start, end) = self.offsets.start_end_unchecked(i);
        let length = end - start;

        // SAFETY: the invariant of the struct
        self.values.sliced_unchecked(start, length)
    }

    /// The optional validity.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    /// The offsets [`Buffer`].
    #[inline]
    pub fn offsets(&self) -> &OffsetsBuffer<O> {
        &self.offsets
    }

    /// The lengths [`Buffer`].
    #[inline]
    pub fn lengths(&self) -> &OffsetsBuffer<O> {
        &self.lengths
    }

    /// The values.
    #[inline]
    pub fn values(&self) -> &Box<dyn Array> {
        &self.values
    }
}

impl<O: Offset> ListViewArray<O> {
    /// Returns a default [`ArrowDataType`]: inner field is named "item" and is nullable
    pub fn default_datatype(data_type: ArrowDataType) -> ArrowDataType {
        let field = Box::new(Field::new("item", data_type, true));
        if O::IS_LARGE {
            ArrowDataType::LargeList(field)
        } else {
            ArrowDataType::List(field)
        }
    }

    /// Returns a the inner [`Field`]
    /// # Panics
    /// Panics iff the logical type is not consistent with this struct.
    pub fn get_child_field(data_type: &ArrowDataType) -> &Field {
        Self::try_get_child(data_type).unwrap()
    }

    /// Returns a the inner [`Field`]
    /// # Errors
    /// Panics iff the logical type is not consistent with this struct.
    pub fn try_get_child(data_type: &ArrowDataType) -> PolarsResult<&Field> {
        if O::IS_LARGE {
            match data_type.to_logical_type() {
                ArrowDataType::LargeList(child) => Ok(child.as_ref()),
                _ => polars_bail!(ComputeError: "ListViewArray<i64> expects DataType::LargeList"),
            }
        } else {
            match data_type.to_logical_type() {
                ArrowDataType::List(child) => Ok(child.as_ref()),
                _ => polars_bail!(ComputeError: "ListViewArray<i32> expects DataType::List"),
            }
        }
    }

    /// Returns a the inner [`ArrowDataType`]
    /// # Panics
    /// Panics iff the logical type is not consistent with this struct.
    pub fn get_child_type(data_type: &ArrowDataType) -> &ArrowDataType {
        Self::get_child_field(data_type).data_type()
    }
}

impl<O: Offset> Array for ListViewArray<O> {
    impl_common_array!();

    fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    #[inline]
    fn with_validity(&self, validity: Option<Bitmap>) -> Box<dyn Array> {
        Box::new(self.clone().with_validity(validity))
    }
}

impl<O: Offset> Splitable for ListViewArray<O> {
    fn check_bound(&self, offset: usize) -> bool {
        offset <= self.len()
    }

    unsafe fn _split_at_unchecked(&self, offset: usize) -> (Self, Self) {
        let (lhs_offsets, rhs_offsets) = unsafe { self.offsets.split_at_unchecked(offset) };
        let (lhs_lengths, rhs_lengths) = unsafe { self.lengths.split_at_unchecked(offset) };
        let (lhs_validity, rhs_validity) = unsafe { self.validity.split_at_unchecked(offset) };

        (
            Self {
                data_type: self.data_type.clone(),
                offsets: lhs_offsets,
                lengths: lhs_lengths,
                validity: lhs_validity,
                values: self.values.clone(),
            },
            Self {
                data_type: self.data_type.clone(),
                offsets: rhs_offsets,
                lengths: rhs_lengths,
                validity: rhs_validity,
                values: self.values.clone(),
            },
        )
    }
}



