use polars_error::PolarsResult;

use super::super::ffi::ToFfi;
use super::super::Array;
use super::ListViewArray;
use crate::array::FromFfi;
use crate::bitmap::align;
use crate::ffi;
use crate::offset::{Offset, OffsetsBuffer};

unsafe impl<O: Offset> ToFfi for ListViewArray<O> {
    fn buffers(&self) -> Vec<Option<*const u8>> {
        vec![
            self.validity.as_ref().map(|x| x.as_ptr()),
            Some(self.offsets.buffer().storage_ptr().cast::<u8>()),
            Some(self.lengths.buffer().storage_ptr().cast::<u8>()),
        ]
    }

    fn children(&self) -> Vec<Box<dyn Array>> {
        vec![self.values.clone()]
    }

    fn offset(&self) -> Option<usize> {
        let offset = self.offsets.buffer().offset();
        if let Some(bitmap) = self.validity.as_ref() {
            if bitmap.offset() == offset {
                Some(offset)
            } else {
                None
            }
        } else {
            Some(offset)
        }
    }

    fn to_ffi_aligned(&self) -> Self {
        let offset = self.offsets.buffer().offset();

        let validity = self.validity.as_ref().map(|bitmap| {
            if bitmap.offset() == offset {
                bitmap.clone()
            } else {
                align(bitmap, offset)
            }
        });

        Self {
            data_type: self.data_type.clone(),
            lengths: self.lengths.clone(),
            offsets: self.offsets.clone(),
            validity,
            values: self.values.clone(),
        }
    }
}

impl<O: Offset, A: ffi::ArrowArrayRef> FromFfi<A> for ListViewArray<O> {
    unsafe fn try_from_ffi(array: A) -> PolarsResult<Self> {
        let data_type = array.data_type().clone();
        let validity = unsafe { array.validity() }?;
        let offsets = unsafe { array.buffer::<O>(1) }?;
        let lengths = unsafe { array.buffer::<O>(2) }?;  // the sizes buffer should be after the offsets buffer per arrow spec
        let child = unsafe { array.child(0)? };
        let values = ffi::try_from(child)?;

        // assumption that data from FFI is well constructed
        let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets) };
        let lengths = unsafe {OffsetsBuffer::new_unchecked(lengths) };

        Self::try_new(data_type, offsets, lengths, values, validity)
    }
}
