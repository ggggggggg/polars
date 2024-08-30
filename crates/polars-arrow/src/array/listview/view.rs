use std::cmp::Ordering;
use std::fmt::{self, Display, Formatter};
use std::ops::Add;

use bytemuck::{Pod, Zeroable};
use polars_error::*;
use polars_utils::min_max::MinMax;
use polars_utils::nulls::IsNull;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::total_ord::{TotalEq, TotalOrd};

use crate::buffer::Buffer;
use crate::datatypes::PrimitiveType;
use crate::types::NativeType;

/// A view into a buffer.
#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct ListViewElement {
    /// The length of the string/bytes.
    pub length: u32,
    /// The buffer index.
    pub buffer_idx: u32,
    /// The offset into the buffer.
    pub offset: u32,
}

impl fmt::Debug for ListViewElement {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {

            fmt.debug_struct("View")
                .field("length", &self.length)
                .field("buffer_idx", &self.buffer_idx)
                .field("offset", &self.offset)
                .finish()
        
    }
}

impl ListViewElement {

    #[inline]
    pub fn new_from_bytes(bytes: &[u8], buffer_idx: u32, offset: u32) -> Self {
        debug_assert!(bytes.len() <= u32::MAX as usize);
            Self {
                length: bytes.len() as u32,
                buffer_idx,
                offset,
            }
        
    }

    /// Constructs a byteslice from this view.
    ///
    /// # Safety
    /// Assumes that this view is valid for the given buffers.
    pub unsafe fn get_slice_unchecked<'a>(&'a self, buffers: &'a [Buffer<u8>]) -> &'a [u8] {
        unsafe {

                let data = buffers.get_unchecked_release(self.buffer_idx as usize);
                let offset = self.offset as usize;
                data.get_unchecked_release(offset..offset + self.length as usize)
            
        }
    }
}

impl IsNull for ListViewElement {
    const HAS_NULLS: bool = false;
    type Inner = Self;

    fn is_null(&self) -> bool {
        false
    }

    fn unwrap_inner(self) -> Self::Inner {
        self
    }
}

impl Display for ListViewElement {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

unsafe impl Zeroable for ListViewElement {}

unsafe impl Pod for ListViewElement {}

impl Add<Self> for ListViewElement {
    type Output = ListViewElement;

    fn add(self, _rhs: Self) -> Self::Output {
        unimplemented!()
    }
}

impl num_traits::Zero for ListViewElement {
    fn zero() -> Self {
        Default::default()
    }

    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl PartialEq for ListViewElement {
    fn eq(&self, other: &Self) -> bool {
        self.length == other.length
            && self.buffer_idx == other.buffer_idx
            && self.offset == other.offset
    }
}

impl TotalOrd for ListViewElement {
    fn tot_cmp(&self, _other: &Self) -> Ordering {
        unimplemented!()
    }
}

impl TotalEq for ListViewElement {
    fn tot_eq(&self, other: &Self) -> bool {
        self.eq(other)
    }
}

impl MinMax for ListViewElement {
    fn nan_min_lt(&self, _other: &Self) -> bool {
        unimplemented!()
    }

    fn nan_max_lt(&self, _other: &Self) -> bool {
        unimplemented!()
    }
}



fn validate_view<F>(views: &[ListViewElement], buffers: &[Buffer<u8>], validate_bytes: F) -> PolarsResult<()>
where
    F: Fn(&[u8]) -> PolarsResult<()>,
{
    for view in views {
        let len = view.length;

            let data = buffers.get(view.buffer_idx as usize).ok_or_else(|| {
                polars_err!(OutOfBounds: "view index out of bounds\n\nGot: {} buffers and index: {}", buffers.len(), view.buffer_idx)
            })?;

            let start = view.offset as usize;
            let end = start + len as usize;
            let b = data
                .as_slice()
                .get(start..end)
                .ok_or_else(|| polars_err!(OutOfBounds: "buffer slice out of bounds"))?;

            validate_bytes(b)?;
        };
    

    Ok(())
}

pub(super) fn validate_binary_view(views: &[ListViewElement], buffers: &[Buffer<u8>]) -> PolarsResult<()> {
    validate_view(views, buffers, |_| Ok(()))
}

fn validate_utf8(b: &[u8]) -> PolarsResult<()> {
    match simdutf8::basic::from_utf8(b) {
        Ok(_) => Ok(()),
        Err(_) => Err(polars_err!(ComputeError: "invalid utf8")),
    }
}

pub(super) fn validate_utf8_view(views: &[ListViewElement], buffers: &[Buffer<u8>]) -> PolarsResult<()> {
    validate_view(views, buffers, validate_utf8)
}

