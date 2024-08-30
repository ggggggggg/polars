use crate::array::{Array, ListViewArray};
use crate::offset::Offset;

pub(super) fn equal<O: Offset>(lhs: &ListViewArray<O>, rhs: &ListViewArray<O>) -> bool {
    lhs.data_type() == rhs.data_type() && lhs.len() == rhs.len() && lhs.iter().eq(rhs.iter())
}
